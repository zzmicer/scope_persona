"""WebRTC transport for the persona stream, in-process.

Why this exists, measured on the shipping MJPEG-over-WebSocket path at 368x640:

                        WebSocket (MJPEG+PCM16)   WebRTC (H.264+Opus)
    video bitrate       4.47 Mbps                 0.97 Mbps
    frame               32.2 KB                   7.4 KB
    inter-frame p99     1603 ms                   137 ms
    stdev               262 ms                    15.3 ms

The p99 is the point. The generator produces 2s of video at once, so the
WebSocket path ships 32 frames in a 0.4s burst and then goes quiet for 1.6s,
against a fixed 0.35s client jitter buffer -- 0.35s of margin against 1.6s gaps
is what a micro-freeze is made of.

Two things fix that here, and they are separable:

  * **RTP timestamps come from the source media clock.** `publish_video` is
    handed the same `(start_frame + i) / fps` the WebSocket path puts in its own
    header, and that becomes the RTP timestamp. The receiver's jitter buffer
    then paces playback off the *generator's* timeline instead of guessing from
    arrival times, which is what makes a bursty source play smoothly.
  * **The sender paces to that clock** rather than dumping a chunk into the
    encoder the moment it exists.

Frames arrive as raw uint8 RGB, straight from the generator's emit worker --
this pays none of the JPEG encode/decode the standalone bridge did.

Feeds off the same emit worker as the WebSocket path rather than replacing it,
so both transports can run at once and be compared on one session.
"""
import asyncio
import fractions
import json
import os
import threading
import time
import urllib.request

import av
import numpy as np
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamTrack

VIDEO_CLOCK = 90000
AUDIO_RATE = 16000
AUDIO_FRAME = 320                 # 20ms, the Opus packetisation
# Bounded backlog, but the floor is ONE CHUNK. The generator hands over a whole
# 2s chunk at once, so a cap below that silently evicts the front of every chunk
# as it is pushed -- at 1.0s exactly half of each chunk never reached the track
# (8.4 fps delivered out of 16.9 published, with 1s holes). The pacer drains at
# realtime, so steady-state occupancy is well under this and the headroom costs
# nothing; it only has to survive the burst.
QUEUE_S = 3.0


_ice_cache = {"servers": None, "expires": 0.0}
_ice_lock = threading.Lock()


def _cloudflare_ice():
    """Short-lived ICE servers from Cloudflare's TURN API, cached until near expiry.

    Worth the API call because Cloudflare hands back **UDP** relay URLs. A relay
    the pod can only reach over TCP puts every frame through one TCP connection,
    where a single loss stalls the stream head-of-line -- measured here as
    latency spikes, a backed-up queue and ~12% of frames shed. Both peers dial
    out to Cloudflare over UDP instead, so the pod never needs UDP ingress.
    """
    key = os.environ.get("SOULX_TURN_CF_ID")
    token = os.environ.get("SOULX_TURN_CF_TOKEN")
    if not (key and token):
        return None
    ttl = int(os.environ.get("SOULX_TURN_TTL", "86400"))
    req = urllib.request.Request(
        f"https://rtc.live.cloudflare.com/v1/turn/keys/{key}"
        f"/credentials/generate-ice-servers",
        data=json.dumps({"ttl": ttl}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            # Cloudflare's WAF rejects urllib's default agent with error 1010
            # ("banned based on your browser's signature") -- the same request
            # from curl succeeds. Identify as something ordinary.
            "User-Agent": "soulx-liveact/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        body = json.loads(r.read())
    servers = body.get("iceServers") or []
    if not isinstance(servers, list):
        return None
    # refresh an hour early rather than handing a client a credential that dies
    # mid-session
    _ice_cache["expires"] = time.time() + max(ttl - 3600, 60)
    return servers


def ice_servers():
    """ICE config for the browser. Never baked into the source.

    Cloudflare TURN if configured (preferred: UDP relay, rotating credentials),
    else a static `SOULX_TURN_URL`. Static TURN credentials are necessarily
    visible to every client -- that is how TURN works -- so treat them as
    demo-scoped and rotate them.
    """
    with _ice_lock:
        if _ice_cache["servers"] and time.time() < _ice_cache["expires"]:
            return _ice_cache["servers"]
        try:
            servers = _cloudflare_ice()
        except Exception as e:  # noqa: BLE001 -- fall back, do not break the page
            print(f"[webrtc] cloudflare TURN unavailable: {e}", flush=True)
            servers = None
        if servers is None:
            url = os.environ.get("SOULX_TURN_URL")
            servers = []
            if url:
                server = {"urls": url}
                if os.environ.get("SOULX_TURN_USER"):
                    server["username"] = os.environ["SOULX_TURN_USER"]
                    server["credential"] = os.environ.get("SOULX_TURN_PASS", "")
                servers = [server]
            _ice_cache["expires"] = time.time() + 300
        _ice_cache["servers"] = servers
        return servers


def _aiortc_ice():
    """The same relays, as aiortc objects, preferring UDP.

    aiortc MUST relay too. Its only other candidate here is the container's
    private address, which a public relay cannot route back to -- so without
    this the browser allocates a relay, sends to 172.19.x.x, and the connection
    never completes.
    """
    out = []
    for s in ice_servers():
        urls = s.get("urls") or []
        if isinstance(urls, str):
            urls = [urls]
        udp = [u for u in urls if "transport=udp" in u] or [
            u for u in urls if u.startswith("turn")
        ]
        if not udp:
            continue
        out.append(
            RTCIceServer(
                urls=udp[0],
                username=s.get("username"),
                credential=s.get("credential"),
            )
        )
    return out


class _Sink:
    """One peer's view of the stream.

    Bounded and drop-oldest: a slow or stalled client must never turn into
    unbounded memory, and must never become backpressure on the generator.
    """

    def __init__(self, loop, maxlen):
        self.loop = loop
        self.queue = asyncio.Queue()
        self.maxlen = maxlen
        self.dropped = 0

    def push(self, item):
        def _put():
            self.queue.put_nowait(item)
            while self.queue.qsize() > self.maxlen:
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    return

        self.loop.call_soon_threadsafe(_put)


class _PacedTrack(MediaStreamTrack):
    """Emits at the source's media clock.

    The anchor (media time <-> wall time) is taken from the first frame and then
    held, so the stream inherits the generator's own pacing instead of drifting
    on a local counter. A backwards jump in media time means the session
    restarted, which re-anchors -- otherwise every frame after a restart would
    look ancient and get rushed out back-to-back.
    """

    def __init__(self, sink):
        super().__init__()
        self.sink = sink
        self.t0_wall = None
        self.t0_media = None

    async def _paced(self):
        media_t, frame = await self.sink.queue.get()
        if self.t0_wall is None or media_t < self.t0_media:
            self.t0_wall, self.t0_media = time.perf_counter(), media_t
        delay = (self.t0_wall + (media_t - self.t0_media)) - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        return frame


class VideoTrack(_PacedTrack):
    kind = "video"

    async def recv(self):
        return await self._paced()


class AudioTrack(_PacedTrack):
    kind = "audio"

    async def recv(self):
        return await self._paced()


class WebRTCHub:
    """Owns an asyncio loop on its own thread and fans frames out to peers.

    The generator runs in threads and Flask's routes are synchronous, so
    everything crossing into aiortc goes through `call_soon_threadsafe` (frames)
    or `run_coroutine_threadsafe` (signalling). Nothing here may block the
    caller: the video worker that calls `publish_video` is on the critical path
    of the stream.
    """

    def __init__(self, fps):
        self.fps = fps
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_loop, name="webrtc", daemon=True
        )
        self.thread.start()
        self.lock = threading.Lock()
        self.peers = {}          # pc -> (video sink, audio sink)
        self.frames_published = 0

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # -- generator side (called from the emit worker threads) ----------------
    def publish_video(self, arr, start_frame):
        """arr: [T,H,W,C] uint8 RGB, as handed to the WebSocket/HLS workers.

        The array belongs to the generator and is reused, so each frame is
        converted here -- once, shared across peers -- rather than referenced.
        """
        with self.lock:
            sinks = [v for v, _a in self.peers.values()]
        if not sinks:
            return
        for i in range(arr.shape[0]):
            media_t = (start_frame + i) / self.fps
            frame = av.VideoFrame.from_ndarray(arr[i], format="rgb24")
            frame.pts = int(round(media_t * VIDEO_CLOCK))
            frame.time_base = fractions.Fraction(1, VIDEO_CLOCK)
            for sink in sinks:
                sink.push((media_t, frame))
            self.frames_published += 1

    def publish_audio(self, pcm_bytes, start_sample):
        with self.lock:
            sinks = [a for _v, a in self.peers.values()]
        if not sinks:
            return
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        for off in range(0, len(samples) - AUDIO_FRAME + 1, AUDIO_FRAME):
            abs_sample = start_sample + off
            frame = av.AudioFrame.from_ndarray(
                samples[off : off + AUDIO_FRAME].reshape(1, -1),
                format="s16",
                layout="mono",
            )
            frame.sample_rate = AUDIO_RATE
            frame.pts = abs_sample
            frame.time_base = fractions.Fraction(1, AUDIO_RATE)
            for sink in sinks:
                sink.push((abs_sample / AUDIO_RATE, frame))

    # -- signalling (called from Flask routes) -------------------------------
    def offer(self, sdp, sdp_type, timeout=20):
        fut = asyncio.run_coroutine_threadsafe(
            self._offer(sdp, sdp_type), self.loop
        )
        return fut.result(timeout=timeout)

    async def _offer(self, sdp, sdp_type):
        pc = RTCPeerConnection(RTCConfiguration(iceServers=_aiortc_ice()))
        vsink = _Sink(self.loop, int(QUEUE_S * self.fps))
        asink = _Sink(self.loop, int(QUEUE_S * AUDIO_RATE / AUDIO_FRAME))

        @pc.on("connectionstatechange")
        async def on_state():
            print(f"[webrtc] peer {pc.connectionState}", flush=True)
            if pc.connectionState in ("failed", "closed"):
                await self._drop(pc)

        pc.addTrack(VideoTrack(vsink))
        pc.addTrack(AudioTrack(asink))
        with self.lock:
            self.peers[pc] = (vsink, asink)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def _drop(self, pc):
        with self.lock:
            self.peers.pop(pc, None)
        try:
            await pc.close()
        except Exception:  # noqa: BLE001 -- teardown races are not interesting
            pass

    def stats(self):
        with self.lock:
            peers = list(self.peers.values())
        return {
            "peers": len(peers),
            "frames_published": self.frames_published,
            "vq": [v.queue.qsize() for v, _a in peers],
            "dropped": sum(v.dropped for v, _a in peers),
        }

    def close(self):
        for pc in list(self.peers):
            asyncio.run_coroutine_threadsafe(self._drop(pc), self.loop)


def ice_json():
    return json.dumps(ice_servers())
