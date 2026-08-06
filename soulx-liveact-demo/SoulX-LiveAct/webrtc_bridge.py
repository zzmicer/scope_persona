"""WebRTC transport test for the persona stream.

Answers one question with numbers: does WebRTC deliver this stream better than the
MJPEG-over-WebSocket path the demo ships with? Measured on the WebSocket path
first (368x640, through the Cloudflare tunnel):

    4.47 Mbps video, 32.2 KB/frame, every frame a keyframe
    inter-frame arrival p50 11.3ms / p99 1603ms  -- 32 frames in a 0.4s burst,
                                                    then 1.6s of nothing

A *bridge*, deliberately, not a patch to interactive_demo.py: it subscribes to
the existing /ws like any other client, so the live demo keeps running untouched
and this can be killed at any time. The cost is a JPEG decode + H.264 encode hop
that an in-process implementation would not pay -- fine for measuring transport,
not what you would ship.

Two effects get bundled together here and should not be confused when reading
the results:
  * the codec (MJPEG -> H.264) is what cuts bitrate, and only WebRTC gets it
  * the pacing (burst -> steady 16fps) is what smooths delivery, and that is
    achievable on the WebSocket path too, without any of this

    python webrtc_bridge.py --port 8095 --source ws://127.0.0.1:8090/ws
"""
import argparse
import asyncio
import fractions
import io
import json
import os
import struct
import time

import av
import numpy as np
import websockets
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack

VIDEO_CLOCK = 90000          # RTP video clock, fixed by spec
AUDIO_RATE = 16000           # what the demo emits
AUDIO_SAMPLES = 320          # 20ms frames, the Opus default packetisation
# Drop rather than grow an unbounded backlog, but the floor is ONE CHUNK: the
# source hands over a whole 2s chunk at once, so a cap below that evicts the
# front of every chunk as it is pushed. At 1.0 exactly half of each chunk was
# lost before the track ever saw it. Above the floor this is just added latency,
# one-for-one, since the generator outruns realtime slightly and keeps the queue
# occupied -- 4.0 put the WebRTC view a full 4s behind the WebSocket view.
MAX_QUEUE_S = 3.0


class Source:
    """Fans the demo's /ws stream out to the tracks.

    Holds video as decoded RGB and audio as int16, both with the media timestamp
    the server stamped on them, so the tracks can pace off the source clock
    instead of guessing.
    """

    def __init__(self, url, fps):
        self.url = url
        self.fps = fps
        self.video = asyncio.Queue()
        self.audio = asyncio.Queue()
        self.frames_in = 0
        self.bytes_in = 0

    async def run(self):
        while True:
            try:
                async with websockets.connect(self.url, max_size=None) as ws:
                    print(f"[bridge] connected to {self.url}", flush=True)
                    async for msg in ws:
                        if not isinstance(msg, (bytes, bytearray)) or len(msg) < 9:
                            continue
                        mtype, ts = struct.unpack("<Bd", msg[:9])
                        payload = msg[9:]
                        self.bytes_in += len(msg)
                        if mtype == 0:
                            self._put_video(ts, payload)
                        elif mtype == 1:
                            self._put_audio(ts, payload)
            except Exception as exc:  # noqa: BLE001 -- reconnect on anything
                print(f"[bridge] source error: {exc}; retrying in 2s", flush=True)
                await asyncio.sleep(2)

    @staticmethod
    def _trim(queue, cap):
        """Drop oldest until under cap.

        Has to be a loop, not a single get: one websocket message carries a whole
        chunk, so audio arrives ~100 frames at a time and dropping one per put
        loses the race -- the queue grew to 14k frames (~5 min of backlog) before
        this was a loop.
        """
        while queue.qsize() > cap:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _put_video(self, ts, jpeg):
        with av.open(io.BytesIO(jpeg)) as container:
            frame = next(container.decode(video=0))
        self.frames_in += 1
        self.video.put_nowait((ts, frame.to_ndarray(format="rgb24")))
        self._trim(self.video, MAX_QUEUE_S * self.fps)

    def _put_audio(self, ts, pcm):
        samples = np.frombuffer(pcm, dtype=np.int16)
        # split into 20ms frames so the Opus encoder gets what it expects
        for i in range(0, len(samples) - AUDIO_SAMPLES + 1, AUDIO_SAMPLES):
            self.audio.put_nowait((ts + i / AUDIO_RATE,
                                   samples[i : i + AUDIO_SAMPLES].copy()))
        self._trim(self.audio, MAX_QUEUE_S * (AUDIO_RATE / AUDIO_SAMPLES))


class PacedVideoTrack(MediaStreamTrack):
    """Emits at a steady fps regardless of how bursty the source was.

    This is the whole smoothness argument: the generator hands over 2s of video
    at once, and something has to spread it back out. Here it is the sender's
    clock; on the WebSocket path it would have to be the emit worker or the
    browser's jitter buffer.
    """

    kind = "video"

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.t0 = None
        self.n = 0

    async def recv(self):
        ts, arr = await self.source.video.get()
        if self.t0 is None:
            self.t0 = time.perf_counter()
        target = self.t0 + self.n / self.source.fps
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = int(self.n * VIDEO_CLOCK / self.source.fps)
        frame.time_base = fractions.Fraction(1, VIDEO_CLOCK)
        self.n += 1
        return frame


class AudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.n = 0

    async def recv(self):
        _ts, samples = await self.source.audio.get()
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = AUDIO_RATE
        frame.pts = self.n * AUDIO_SAMPLES
        frame.time_base = fractions.Fraction(1, AUDIO_RATE)
        self.n += AUDIO_SAMPLES
        return frame


PAGE = """<!doctype html><meta charset=utf-8><title>WebRTC transport test</title>
<style>body{background:#111;color:#eee;font:14px system-ui;text-align:center}
video{max-height:80vh;border-radius:8px}pre{text-align:left;display:inline-block}</style>
<h3>WebRTC transport test</h3><video id=v autoplay playsinline></video><pre id=s></pre>
<script>
const pc = new RTCPeerConnection({iceServers: __ICE__});
pc.addTransceiver('video', {direction:'recvonly'});
pc.addTransceiver('audio', {direction:'recvonly'});
pc.ontrack = e => { document.getElementById('v').srcObject = e.streams[0]; };
(async () => {
  const offer = await pc.createOffer(); await pc.setLocalDescription(offer);
  await new Promise(r => pc.iceGatheringState === 'complete' ? r()
      : pc.addEventListener('icegatheringstatechange',
          () => pc.iceGatheringState === 'complete' && r()));
  const res = await fetch('/offer', {method:'POST',
      headers:{'content-type':'application/json'},
      body: JSON.stringify({sdp: pc.localDescription.sdp, type: pc.localDescription.type})});
  await pc.setRemoteDescription(await res.json());
  setInterval(async () => {
    const stats = await pc.getStats(); let out = '';
    stats.forEach(r => { if (r.type === 'inbound-rtp' && r.kind === 'video')
      out = `bytes ${r.bytesReceived}\\nframes ${r.framesDecoded}\\n` +
            `dropped ${r.framesDropped}\\njitter ${r.jitter}`; });
    document.getElementById('s').textContent = out;
  }, 1000);
})();
</script>"""


def ice_servers():
    """TURN config from the environment, never baked into the source.

    A pod with no UDP ingress (RunPod maps TCP ports only) cannot do direct
    WebRTC at all: the browser has to reach a TURN server over TCP, which then
    relays to aiortc locally. Credentials are static and necessarily visible to
    the client -- that is how TURN works -- so treat them as demo-scoped and
    rotate them rather than assuming the page keeps them secret.
    """
    url = os.environ.get("SOULX_TURN_URL")
    if not url:
        return []
    server = {"urls": url}
    if os.environ.get("SOULX_TURN_USER"):
        server["username"] = os.environ["SOULX_TURN_USER"]
        server["credential"] = os.environ.get("SOULX_TURN_PASS", "")
    return [server]


async def index(_request):
    return web.Response(
        content_type="text/html",
        text=PAGE.replace("__ICE__", json.dumps(ice_servers())),
    )


async def offer(request):
    params = await request.json()
    pc = RTCPeerConnection()
    request.app["pcs"].add(pc)

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"[bridge] pc {pc.connectionState}", flush=True)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            request.app["pcs"].discard(pc)

    src = request.app["source"]
    pc.addTrack(PacedVideoTrack(src))
    pc.addTrack(AudioTrack(src))

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


async def stats(request):
    """Source counters plus per-peer RTP bytes.

    Bitrate has to come from the sender: aiortc's RTCInboundRtpStreamStats
    carries packetsReceived but no bytesReceived, so a receiving probe cannot
    measure it. RTCOutboundRtpStreamStats.bytesSent is the real wire number.
    """
    src = request.app["source"]
    sent = {"video": 0, "audio": 0, "packets": 0}
    for pc in request.app["pcs"]:
        for report in (await pc.getStats()).values():
            if report.type == "outbound-rtp":
                sent[report.kind] = sent.get(report.kind, 0) + report.bytesSent
                sent["packets"] += report.packetsSent
    return web.json_response(
        {"frames_in": src.frames_in, "bytes_in": src.bytes_in,
         "vq": src.video.qsize(), "aq": src.audio.qsize(),
         "peers": len(request.app["pcs"]), "sent": sent}
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--source", default="ws://127.0.0.1:8090/ws")
    ap.add_argument("--fps", type=int, default=16)
    args = ap.parse_args()

    app = web.Application()
    app["pcs"] = set()
    app["source"] = Source(args.source, args.fps)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_get("/stats", stats)

    async def start_source(app):
        app["task"] = asyncio.ensure_future(app["source"].run())

    app.on_startup.append(start_source)
    print(f"[bridge] webrtc test on http://0.0.0.0:{args.port} "
          f"<- {args.source}", flush=True)
    web.run_app(app, host="0.0.0.0", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
