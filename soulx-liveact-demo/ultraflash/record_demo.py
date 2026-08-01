"""Record a fixed-length action demo off the live /ws stream and mux to mp4.

The stream is binary-framed: struct("<Bd", mtype, ts) + payload, where
mtype 0 = JPEG frame (ts = frame_index/fps) and mtype 1 = PCM16 (ts =
start_sample/SR). A JSON meta message with fps/sr/w/h arrives first.

Actions are scheduled by FRAME COUNT, not wall clock, so the resulting video has
identical content at every resolution even though the slower ones generate below
realtime. That makes the clips directly comparable.
"""

import argparse
import json
import os
import queue
import struct
import subprocess
import threading
import time

import requests
import websocket

ACTIONS = [
    (8,   "She waves her hand at the camera cheerfully!"),
    (88,  "She rests her chin on both hands and smiles warmly."),
    (168, "She tilts her head to the side and winks playfully."),
    (248, "She stretches her arms and looks away thoughtfully."),
]
SAYS = [
    (40,  "Hi there! Great to finally meet you."),
    (200, "Want to see what else I can do?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--no_say", action="store_true")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    fdir = os.path.join(args.workdir, "frames")
    os.makedirs(fdir, exist_ok=True)
    for f in os.listdir(fdir):
        os.remove(os.path.join(fdir, f))

    ws = websocket.create_connection(f"ws://127.0.0.1:{args.port}/ws", timeout=120)
    meta = json.loads(ws.recv())
    fps, sr = int(meta["fps"]), int(meta["sr"])
    want = int(args.seconds * fps)
    print(f"meta {meta}  -> recording {want} frames", flush=True)

    # Fire directives from a side thread so the socket keeps draining; a blocked
    # reader would let the server's per-client queue drop frames.
    fired = set()
    pending = queue.Queue()

    def sender():
        while True:
            item = pending.get()
            if item is None:
                return
            kind, text = item
            try:
                requests.post(f"{base}/{kind}", json={"text": text}, timeout=10)
                print(f"  [{kind}] {text}", flush=True)
            except Exception as e:
                print(f"  [{kind}] FAILED {e}", flush=True)

    th = threading.Thread(target=sender, daemon=True)
    th.start()

    n_frames = 0
    audio = bytearray()
    t0 = time.time()
    while n_frames < want:
        buf = ws.recv()
        if isinstance(buf, str):
            continue
        mtype, ts = struct.unpack_from("<Bd", buf, 0)
        payload = buf[9:]
        if mtype == 0:
            with open(os.path.join(fdir, f"{n_frames:06d}.jpg"), "wb") as fh:
                fh.write(payload)
            n_frames += 1
            for idx, text in ACTIONS:
                if n_frames >= idx and ("a", idx) not in fired:
                    fired.add(("a", idx))
                    pending.put(("action", text))
            if not args.no_say:
                for idx, text in SAYS:
                    if n_frames >= idx and ("s", idx) not in fired:
                        fired.add(("s", idx))
                        pending.put(("say", text))
            if n_frames % 32 == 0:
                print(f"  {n_frames}/{want} frames  ({time.time()-t0:.0f}s wall)", flush=True)
        elif mtype == 1:
            audio += payload

    pending.put(None)
    ws.close()
    wall = time.time() - t0
    print(f"captured {n_frames} frames + {len(audio)} audio bytes in {wall:.1f}s wall "
          f"({n_frames/fps:.1f}s of video -> {n_frames/fps/wall:.2f}x realtime)", flush=True)

    apcm = os.path.join(args.workdir, "audio.pcm")
    with open(apcm, "wb") as fh:
        fh.write(audio)

    # Silence-only capture still muxes fine; only a genuinely empty one is dropped.
    has_audio = len(audio) > sr  # >0.5s of int16
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(fdir, "%06d.jpg")]
    if has_audio:
        cmd += ["-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", apcm]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    cmd += [args.out]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg FAILED:\n" + r.stderr[-2000:], flush=True)
        return 1
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
