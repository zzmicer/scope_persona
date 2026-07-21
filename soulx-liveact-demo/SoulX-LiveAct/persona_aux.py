"""Isolated Qwen + Kokoro service for the interactive SoulX demo."""

import argparse
import threading

import numpy as np
import torch
import torchaudio.transforms as T
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer

SR = 16000
app = Flask(__name__)
lock = threading.Lock()
tokenizer = None
model = None
tts_pipe = None
resampler = None


def ensure_brain():
    global tokenizer, model
    if model is None:
        name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = (
            AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
            .to("cuda:0")
            .eval()
        )


def ensure_tts():
    global tts_pipe, resampler
    if tts_pipe is None:
        from kokoro import KPipeline

        tts_pipe = KPipeline(lang_code="a", device="cuda:0")
        resampler = T.Resample(24000, SR)


def preload():
    with lock:
        ensure_brain()
        ensure_tts()
    print("aux models ready", flush=True)


@app.get("/health")
def health():
    return jsonify({"ok": True, "ready": model is not None and tts_pipe is not None})


@app.post("/generate")
def generate():
    messages = request.get_json(force=True).get("messages") or []
    with lock:
        ensure_brain()
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **ids,
                max_new_tokens=160,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            output[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
        )
    return jsonify({"text": raw})


@app.post("/tts")
def tts():
    payload = request.get_json(force=True)
    text = (payload.get("text") or "").strip()
    voice = payload.get("voice") or "af_heart"
    with lock:
        ensure_tts()
        chunks = [
            audio for _, _, audio in tts_pipe(text, voice=voice) if audio is not None
        ]
        if not chunks:
            wav = np.zeros(0, dtype=np.float32)
        else:
            wav24 = torch.cat(chunks).float().cpu()
            wav = resampler(wav24.unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
    return app.response_class(wav.tobytes(), mimetype="application/octet-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()
    if args.preload:
        threading.Thread(target=preload, daemon=True).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)
