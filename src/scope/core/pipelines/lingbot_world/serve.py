"""Minimal web UI for the LingBot-World-V2 interactive session.

Single-page app: upload a start image + world prompt, then steer the world
with text commands (motion or events). Each command generates ~2s of video
in ~10-20s on one H200; the page shows the latest frame and the growing
session video.

    python serve.py --lingbot-repo /workspace/lingbot-world-v2 \
        --ckpt-dir /workspace/lingbot-world-v2-14b-causal-fast --port 8189

Not for public deployment: no auth, one global session, blocking turns.
"""

import argparse
import io
import logging
import os
import sys
import tempfile
import threading
import time

_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>LingBot World — interactive demo</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background:#0b0f14; color:#e6edf3; }
.wrap { max-width: 900px; margin: 0 auto; padding: 16px; }
h1 { font-size: 18px; font-weight: 600; }
.card { background:#151b23; border:1px solid #2d333b; border-radius:10px; padding:14px; margin-bottom:14px; }
label { display:block; font-size:12px; color:#9ea7b3; margin:8px 0 4px; }
input[type=text], textarea { width:100%; box-sizing:border-box; background:#0b0f14; color:#e6edf3; border:1px solid #2d333b; border-radius:6px; padding:8px; font-size:14px; }
button { background:#2f81f7; color:white; border:0; border-radius:6px; padding:9px 16px; font-size:14px; cursor:pointer; margin-top:8px; }
button:disabled { background:#30363d; cursor:wait; }
#frame { width:100%; border-radius:8px; display:none; }
video { width:100%; border-radius:8px; margin-top:10px; display:none; }
#log { font-family: ui-monospace, monospace; font-size:12px; white-space:pre-wrap; color:#9ea7b3; max-height:180px; overflow-y:auto; }
.hint { font-size:12px; color:#768390; margin-top:6px; }
.status { font-size:13px; color:#3fb950; margin-left:10px; }
</style></head><body><div class="wrap">
<h1>🌍 LingBot-World-V2 — control the world with text</h1>

<div class="card" id="startCard">
  <label>Start image (the world + your character)</label>
  <input type="file" id="image" accept="image/*">
  <label>World prompt (describe the scene and the character; say they are alone)</label>
  <textarea id="prompt" rows="3">A cheerful young woman with a baseball cap and black rain jacket stands alone on a rocky mountain trail, smiling at the camera. She is the only person on the trail. Behind her, a vast glacial river valley winds between green and rust-colored mountains under a dramatic cloudy sky.</textarea>
  <button id="startBtn" onclick="startSession()">Start session</button><span class="status" id="startStatus"></span>
</div>

<div class="card" id="cmdCard" style="display:none">
  <label>Command — motion: walk forward / turn left / orbit around her / look up / stay ·
  anything else is an event: "she waves at the camera"</label>
  <input type="text" id="cmd" placeholder="walk forward for 2 seconds" onkeydown="if(event.key==='Enter')sendCmd()">
  <button id="cmdBtn" onclick="sendCmd()">Send</button><span class="status" id="cmdStatus"></span>
  <div class="hint" id="budget"></div>
  <img id="frame">
  <video id="vid" controls muted></video>
</div>

<div class="card"><div id="log">ready — start a session\n</div></div>

<script>
const log = (m) => { const el = document.getElementById('log'); el.textContent += m + "\\n"; el.scrollTop = el.scrollHeight; };
async function startSession() {
  const f = document.getElementById('image').files[0];
  if (!f) { alert('pick an image'); return; }
  const fd = new FormData();
  fd.append('image', f);
  fd.append('prompt', document.getElementById('prompt').value);
  document.getElementById('startBtn').disabled = true;
  document.getElementById('startStatus').textContent = 'encoding image… (~15s)';
  log('> start session');
  const r = await fetch('/api/start', { method:'POST', body: fd });
  const j = await r.json();
  document.getElementById('startBtn').disabled = false;
  document.getElementById('startStatus').textContent = '';
  if (!r.ok) { log('! ' + j.detail); return; }
  log('session ready: ' + j.width + 'x' + j.height + ', budget ~' + j.budget_seconds + 's of video');
  document.getElementById('cmdCard').style.display = 'block';
  document.getElementById('budget').textContent = 'budget left: ~' + j.budget_seconds + 's';
}
async function sendCmd() {
  const cmd = document.getElementById('cmd').value.trim();
  if (!cmd) return;
  document.getElementById('cmdBtn').disabled = true;
  document.getElementById('cmdStatus').textContent = 'generating… (~10-20s)';
  log('> ' + cmd);
  const r = await fetch('/api/command', { method:'POST',
    headers: {'Content-Type':'application/json'}, body: JSON.stringify({text: cmd}) });
  const j = await r.json();
  document.getElementById('cmdBtn').disabled = false;
  document.getElementById('cmdStatus').textContent = '';
  if (!r.ok) { log('! ' + j.detail); return; }
  log('  ' + j.kind + ' → +' + j.new_frames + ' frames (' + j.video_seconds.toFixed(1) + 's) in ' + j.gen_seconds.toFixed(1) + 's');
  document.getElementById('budget').textContent = 'budget left: ~' + j.budget_seconds_left.toFixed(0) + 's';
  const img = document.getElementById('frame');
  img.src = '/api/frame.jpg?t=' + Date.now(); img.style.display = 'block';
  const v = document.getElementById('vid');
  v.src = '/api/video?t=' + Date.now(); v.style.display = 'block';
  document.getElementById('cmd').value = '';
}
</script>
</div></body></html>"""


def build_app(pipe, args):
    import numpy as np
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, Response
    from PIL import Image

    from scope.core.pipelines.lingbot_world.actions import (
        Motion,
        parse_command,
        trajectory_from_motions,
    )
    from scope.core.pipelines.lingbot_world.session import LingbotWorldSession

    app = FastAPI(title="lingbot-world demo")
    state = {"session": None, "base_prompt": "", "prompt_is_event": False}
    lock = threading.Lock()
    out_path = os.path.join(tempfile.gettempdir(), "lingbot_ui_session.mp4")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _HTML

    @app.post("/api/start")
    def start(image: UploadFile = File(...), prompt: str = Form(...)):
        with lock:
            img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
            state["session"] = LingbotWorldSession(
                pipe,
                img,
                prompt,
                max_frames=args.max_frames,
                chunk_size=args.chunk_size,
                seed=args.seed,
            )
            state["base_prompt"] = prompt
            state["prompt_is_event"] = False
            s = state["session"]
            return {
                "width": s.w,
                "height": s.h,
                "budget_seconds": s.lat_f_max // 4,
            }

    @app.post("/api/command")
    def command(body: dict):
        with lock:
            s = state["session"]
            if s is None:
                raise HTTPException(400, "no session — start one first")
            if s.latent_budget_left == 0:
                raise HTTPException(
                    400, "latent budget exhausted — start a new session"
                )
            text = str(body.get("text", "")).strip()
            if not text:
                raise HTTPException(400, "empty command")

            motions, event = parse_command(text)
            if event is not None:
                s.set_prompt(f"{state['base_prompt']} {event}")
                state["prompt_is_event"] = True
                motions = [Motion(kind="stay", seconds=2.0)]
                kind = f"event: {event!r}"
            else:
                if state["prompt_is_event"]:
                    s.set_prompt(state["base_prompt"])
                    state["prompt_is_event"] = False
                kind = "motion: " + "+".join(m.kind for m in motions)

            track = trajectory_from_motions(motions, s.cur_c2w, chunk_size=s.chunk_size)
            track = track[: s.latent_budget_left]
            if len(track) == 0:
                raise HTTPException(400, "latent budget exhausted")

            t0 = time.perf_counter()
            frames = s.step(track)
            dt = time.perf_counter() - t0
            s.save(out_path)

            last = frames[:, -1]
            arr = ((last.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(
                np.uint8
            )
            state["last_frame"] = Image.fromarray(arr)

            return {
                "kind": kind,
                "new_frames": int(frames.shape[1]),
                "video_seconds": frames.shape[1] / 16.0,
                "gen_seconds": dt,
                "budget_seconds_left": s.latent_budget_left / 4.0,
            }

    @app.get("/api/frame.jpg")
    def frame():
        img = state.get("last_frame")
        if img is None:
            raise HTTPException(404, "no frames yet")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return Response(buf.getvalue(), media_type="image/jpeg")

    @app.get("/api/video")
    def video():
        if not os.path.exists(out_path):
            raise HTTPException(404, "no video yet")
        return FileResponse(out_path, media_type="video/mp4")

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lingbot-repo", default=os.environ.get("LINGBOT_WORLD_REPO"))
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--port", type=int, default=8189)
    p.add_argument("--max-frames", type=int, default=321)
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--local-attn-size", type=int, default=18)
    p.add_argument("--sink-size", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    assert args.lingbot_repo, "--lingbot-repo or LINGBOT_WORLD_REPO required"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    sys.path.insert(0, args.lingbot_repo)

    import uvicorn
    import wan
    from wan.configs import WAN_CONFIGS

    logging.info("loading pipeline (~1 min)...")
    pipe = wan.WanI2VCausal(
        config=WAN_CONFIGS["i2v-A14B"],
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        init_on_cpu=False,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        infer_mode="causal_fast",
    )
    logging.info("pipeline loaded; serving on :%d", args.port)

    uvicorn.run(build_app(pipe, args), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
