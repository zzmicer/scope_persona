"""Interactive persona demo for SoulX-LiveAct.

A continuous live session driven by chat, modeled on Wan-Streamer v0.3's
"world + event stream": the T5 context is composed as world (persistent
identity/scene) + sustained state (held posture) + a transient transition.
Transitions are held a few chunks then dropped, but posture changes update the
sustained state so the character STAYS in the new pose (sit -> keeps sitting).

  - /chat    -> LLM decides {say, action, pose}; speech -> kokoro TTS (lip-synced);
               action = transient motion, pose = new sustained state (or null).
  - /say     -> speak exactly this text.
  - /action  -> perform a motion now; {pose} or {persist:true} makes it stick,
               empty body clears back to neutral idle.
Video+audio are muxed into a live HLS stream at /stream/live/live.m3u8.

Run (2 GPUs):
  USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=2,3 \
  torchrun --nproc_per_node=2 --master_port=29617 interactive_demo.py \
    --ckpt_dir /workspace/soulx/weights/LiveAct \
    --wav2vec_dir /workspace/soulx/weights/chinese-wav2vec2-base \
    --size 720*416 --fps 20 --port 8090 \
    --image /workspace/soulx_setup/chano39-Anime-Original-anime-9101906.png
"""

import os
import argparse
import threading
import time
import subprocess
import shutil
import json
import gc
import concurrent.futures as _cf
import queue
import urllib.request

import io
import struct
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
import torchaudio.transforms as T
from torchvision import transforms
from PIL import Image
from flask import (
    Flask,
    send_file,
    send_from_directory,
    jsonify,
    request,
    render_template,
)
from flask_sock import Sock

from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
from util_liveact import (
    center_rescale_crop_keep_ratio,
    get_embedding,
    get_msk,
    get_audio_emb,
)
from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
from fp8_gemm import FP8GemmOptions, enable_fp8_gemm

gc.collect()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

app = Flask(__name__)
sock = Sock(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HLS_ROOT = os.path.join(BASE_DIR, "hls_output")
M3U8_NAME = "live.m3u8"
os.makedirs(HLS_ROOT, exist_ok=True)

SR = 16000  # everything audio inside the session is 16k mono float32

# Wan-Streamer v0.3 "world + event stream": the world prompt is the PERSISTENT,
# pose-neutral identity/scene. It must NOT lock a posture (e.g. "looks at the
# camera"), or reverting to it would pull the character back out of any sustained
# pose (sitting, turned around, ...). Held posture lives in the sticky state
# prompt + the rolling KV cache, not here.
WORLD_PROMPT = (
    "A beautiful blonde anime girl with an expressive, friendly face, "
    "in a cozy warm room with soft natural lighting, gentle ambient motion."
)
DEFAULT_PERSONA = "Cheerful, warm, playful, curious, and wholesome."
KOKORO_VOICES = {"af_heart", "af_bella", "af_nicole", "am_adam"}


# Latent grid = (H // 8) x (W // 8) further patchified by 2, so both dimensions
# must be multiples of 16; anything else fails much later with an opaque reshape
# error deep inside the transformer. Orientation itself is free: the model, the
# KV cache and the SP sharding (which splits the FRAME axis, not space) only see
# frame_len = (H/16)*(W/16), so 416*720 costs exactly what 720*416 costs.
# one worker only: the sidecar serializes on its own lock, and two chunks in
# flight would reorder through its streaming caches
_sr_pool = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sr")

SIZE_ALIGN = 16


def _sr_request(base, latent, ctx, rank, timeout=60):
    """POST one chunk latent to the SR sidecar, return HR uint8 [T,H,W,C]."""
    import io as _io

    import numpy as _np
    import requests as _rq

    if rank != 0:
        return None
    buf = _io.BytesIO()
    torch.save(
        {
            "latent": latent.detach().to(torch.bfloat16).cpu(),
            "ctx": [c.detach().to(torch.bfloat16).cpu() for c in ctx],
        },
        buf,
    )
    r = _rq.post(f"{base}/sr", data=buf.getvalue(), timeout=timeout)
    r.raise_for_status()
    hdr, _, payload = r.content.partition(b"\n")
    t, h, w, c = (int(x) for x in hdr.decode().split(","))
    return _np.frombuffer(payload, dtype=_np.uint8).reshape(t, h, w, c)


def _sr_reset(base):
    try:
        import requests as _rq

        _rq.post(f"{base}/reset", timeout=30)
    except Exception as e:
        print(f"[sr] reset failed: {e}", flush=True)
_DUMP_DIR = os.environ.get("SOULX_DUMP_LATENTS", "")
_DUMP_N = int(os.environ.get("SOULX_DUMP_N", "12"))


def _parse_size(size):
    """'W*H' -> (width, height), validated. Portrait and landscape both allowed."""
    try:
        w, h = (int(x) for x in size.split("*"))
    except ValueError:
        raise SystemExit(f"--size must look like WIDTH*HEIGHT (got {size!r})") from None
    bad = [n for n, v in (("width", w), ("height", h)) if v <= 0 or v % SIZE_ALIGN]
    if bad:
        raise SystemExit(
            f"--size {size}: {' and '.join(bad)} must be a positive multiple of "
            f"{SIZE_ALIGN} (e.g. 720*416 landscape, 416*720 portrait, 480*832 tall)"
        )
    return w, h


# vLLM's W8A8 wrapper replaces EVERY nn.Linear it is handed. In a diffusion
# transformer the small conditioning layers carry almost no FLOPs but are very
# sensitive to quantization error, and wrapping them is the likely source of the
# noise that got FP8 switched off on this pod:
#   time_projection  -> the AdaLN modulation that scales every block's residual
#   time/text_embedding, img_emb (CLIP identity), audio_proj (lipsync)
#   head             -> the final velocity prediction
# scope="blocks" keeps FP8 on the attention/FFN matmuls, which hold essentially
# all the compute, and leaves the above in bf16. scope="all" is the old
# behaviour, kept so the two can be compared.
_FP8_BLOCK_PARTS = {"self_attn", "cross_attn", "ffn"}


def _fp8_block_matmuls(name, module):
    # names look like "blocks.12.self_attn.q" / "blocks.12.ffn.0"; match on the
    # path component so "audio_cross_attn" (lipsync) is not caught by substring
    parts = name.split(".")
    return len(parts) > 2 and parts[0] == "blocks" and parts[2] in _FP8_BLOCK_PARTS


def _prune_uploads(dirpath, keep=5):
    """Keep only the newest `keep` uploaded references (names are unique)."""
    try:
        files = sorted(
            (os.path.join(dirpath, n) for n in os.listdir(dirpath)),
            key=os.path.getmtime,
            reverse=True,
        )
        for stale in files[keep:]:
            os.remove(stale)
    except OSError:
        pass  # best effort; a full uploads dir must never break a session start


def _clean_field(v):
    """Normalize an LLM JSON string field -> stripped str or None."""
    if not isinstance(v, str):
        return None
    v = v.strip()
    return None if v.lower() in ("", "null", "none") else v


# ----------------------------------------------------------------------------
# rank0 session state (fed by Flask routes, consumed at chunk boundaries)
# ----------------------------------------------------------------------------
class SessionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.speech_q = queue.Queue()  # np.float32 arrays @16k queued for playback
        # structured directive picked up at the next chunk boundary:
        #   {"transition": str|None, "rest": str|None}
        # transition = transient motion held for action_hold chunks then dropped.
        # rest = new sustained state; None leaves state unchanged (gesture),
        #        "" clears it back to neutral idle.
        self.pending_directive = None
        self.stop_flag = False
        self.start_params = queue.Queue()  # session start requests
        self.active = False
        self.chunk_count = 0
        self.last_error = None
        self.events = []  # rolling log for the UI
        self.persona_name = "Chano"
        self.persona_prompt = DEFAULT_PERSONA
        self.voice = "af_heart"

    def log(self, kind, text):
        with self.lock:
            self.events.append({"t": time.time(), "kind": kind, "text": text})
            self.events = self.events[-100:]


STATE = SessionState()

# ----------------------------------------------------------------------------
# WebSocket frame feed: binary msgs = 1B type (0=jpeg,1=pcm16@16k) + f64le ts + payload
# ----------------------------------------------------------------------------
WS_CLIENTS = set()
WS_LOCK = threading.Lock()


def ws_broadcast(msg):
    with WS_LOCK:
        clients = list(WS_CLIENTS)
    for q in clients:
        try:
            q.put_nowait(msg)
        except queue.Full:
            try:  # drop oldest, keep stream realtime for slow clients
                q.get_nowait()
                q.put_nowait(msg)
            except Exception:
                pass


def ws_pack(mtype, ts, payload):
    return struct.pack("<Bd", mtype, ts) + payload


# ----------------------------------------------------------------------------
# TTS (kokoro, lazy)
# ----------------------------------------------------------------------------
class TTS:
    def __init__(self, device, remote_url=None):
        self.device = f"cuda:{device}" if str(device).isdigit() else str(device)
        self.remote_url = remote_url.rstrip("/") if remote_url else None
        self.pipe = None
        self.resampler = None

    def ensure(self):
        if self.pipe is None:
            from kokoro import KPipeline

            self.pipe = KPipeline(lang_code="a", device=self.device)
            self.resampler = T.Resample(24000, SR)

    def synth(self, text, voice="af_heart"):
        """text -> np.float32 mono @16k"""
        if self.remote_url:
            req = urllib.request.Request(
                f"{self.remote_url}/tts",
                data=json.dumps({"text": text, "voice": voice}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                return np.frombuffer(response.read(), dtype=np.float32).copy()
        self.ensure()
        chunks = []
        for _, _, audio in self.pipe(text, voice=voice):
            if audio is not None:
                chunks.append(audio)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        wav24 = torch.cat(chunks).float().cpu()
        wav16 = self.resampler(wav24.unsqueeze(0)).squeeze(0)
        return wav16.numpy().astype(np.float32)


# ----------------------------------------------------------------------------
# LLM brain (lazy; small instruct model). Falls back to echo mode.
# ----------------------------------------------------------------------------
class Brain:
    def __init__(self, device, remote_url=None):
        self.device = f"cuda:{device}" if str(device).isdigit() else str(device)
        self.remote_url = remote_url.rstrip("/") if remote_url else None
        self.model = None
        self.tok = None
        self.failed = False
        self.history = []
        self.lock = threading.Lock()

    def _system_prompt(self):
        return (
            f"You are {STATE.persona_name}, an anime character appearing live on video. "
            f"Your personality is: {STATE.persona_prompt} "
            "Chat naturally with the user, in English, using 1-2 short sentences. "
            "You can also perform physical motions on camera. "
            'Reply ONLY with JSON: {"say": "<what you say>", '
            '"action": "<short third-person visual description of the MOTION you make now, '
            "e.g. 'She waves her hand at the camera cheerfully!' or null if no special motion>\", "
            '"pose": "<if that motion changes your sustained body posture or position '
            "(sit down, stand up, lie down, turn around, lean back, kneel, cross arms), a SHORT "
            "third-person description of the RESULTING held pose you stay in afterwards, "
            "e.g. 'She is sitting on the floor, relaxed.' — otherwise null for momentary "
            'gestures (wave, nod, wink, laugh) that should not persist>"}. '
            "Keep everything friendly and wholesome."
        )

    def reset_history(self):
        with self.lock:
            self.history = []

    def ensure(self):
        if self.model is None and not self.failed:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                name = os.environ.get("PERSONA_LLM", "Qwen/Qwen2.5-1.5B-Instruct")
                self.tok = AutoTokenizer.from_pretrained(name)
                dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
                self.model = (
                    AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
                    .to(self.device)
                    .eval()
                )
            except Exception as e:
                print(f"[Brain] LLM unavailable, echo mode: {e}", flush=True)
                self.failed = True

    def reply(self, message):
        """-> (say, action)"""
        with self.lock:
            self.history.append({"role": "user", "content": message})
            msgs = [
                {"role": "system", "content": self._system_prompt()}
            ] + self.history[-12:]
            if self.remote_url:
                req = urllib.request.Request(
                    f"{self.remote_url}/generate",
                    data=json.dumps({"messages": msgs}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=120) as response:
                    raw = json.loads(response.read())["text"]
            else:
                self.ensure()
                if self.model is None:
                    return f"You said: {message}", None, None
                text = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                ids = self.tok(text, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **ids,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tok.eos_token_id,
                    )
                raw = self.tok.decode(
                    out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
                )
            say, action, pose = raw.strip(), None, None
            try:
                s = raw[raw.index("{") : raw.rindex("}") + 1]
                obj = json.loads(s)
                say = obj.get("say") or ""
                action = _clean_field(obj.get("action"))
                pose = _clean_field(obj.get("pose"))
            except Exception:
                pass
            self.history.append({"role": "assistant", "content": say})
            return say, action, pose


# ----------------------------------------------------------------------------
# Distributed engine
# ----------------------------------------------------------------------------
class LiveEngine:
    def __init__(self, args):
        self.args = args
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.device = self.local_rank
        self.width, self.height = _parse_size(args.size)
        # SR changes only the OUTPUT size; the generator keeps running at self.width
        # x self.height, so frame_len / KV-cache sizing below must not use these.
        self.sr_url = getattr(args, "sr_url", None)
        self.sr_scale = int(getattr(args, "sr_scale", 2)) if self.sr_url else 1
        self.out_width = self.width * self.sr_scale
        self.out_height = self.height * self.sr_scale
        self.fps = args.fps
        self.use_dist = self.world_size > 1

        if not dist.is_initialized() and self.use_dist:
            torch.cuda.set_device(self.device)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=self.rank,
                world_size=self.world_size,
            )
        self.control_pg = (
            dist.new_group(backend="gloo", timeout=timedelta(hours=48))
            if self.use_dist
            else None
        )

        if self.use_dist:
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )

            init_distributed_environment(rank=self.rank, world_size=self.world_size)
            initialize_model_parallel(
                sequence_parallel_degree=self.world_size,
                ring_degree=1,
                ulysses_degree=self.world_size,
            )

        if self.use_dist:
            from model_liveact.model_memory_sp import WanModel
        else:
            from model_liveact.model_memory import WanModel
        self.wan = WanModel.from_pretrained(
            args.ckpt_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
        ).to(dtype=torch.bfloat16)
        if not args.no_fp8_gemm:
            mf = None if args.fp8_scope == "all" else _fp8_block_matmuls
            enable_fp8_gemm(self.wan, options=FP8GemmOptions(), module_filter=mf)
            wrapped = sum(
                1 for m in self.wan.modules() if type(m).__name__ == "FP8Linear"
            )
            print(
                f"fp8 gemm: scope={args.fp8_scope}, {wrapped} linears wrapped",
                flush=True,
            )
        self.wan = self.wan.to(self.device)
        self.wan.freqs = self.wan.freqs.to(self.device)
        self.wan.eval()
        if args.compile_blocks:
            # Pruna's open-source Wan recipe compiles the transformer ModuleList
            # one block at a time. This is more robust to graph breaks than
            # compiling the stateful streaming WanModel as one graph.
            for i, block in enumerate(self.wan.blocks):
                self.wan.blocks[i] = torch.compile(
                    block,
                    mode=args.compile_mode,
                    backend="inductor",
                    dynamic=False,
                )
            print(
                f"torch.compile: 40 blocks, mode={args.compile_mode}",
                flush=True,
            )
        elif not args.no_compile:
            self.wan = torch.compile(
                self.wan,
                mode="max-autotune-no-cudagraphs",
                backend="inductor",
                dynamic=False,
            )

        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.timesteps = [
            torch.tensor([t]).to(self.device, dtype=torch.float32)
            for t in [1000.0, 937.5, 833.33333333, 0.0]
        ]

        self.transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda im: center_rescale_crop_keep_ratio(
                        im, (self.height, self.width)
                    )
                ),
                transforms.ToTensor(),
                transforms.Resize((self.height, self.width)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.vae = LightVAE(
            vae_path=os.path.join(args.ckpt_dir, "Wan2.1_VAE.pth"),
            dtype=torch.bfloat16,
            device=self.device,
            use_lightvae=False,
            parallel=self.use_dist,
        )
        self.clip = CLIPModel(
            checkpoint_path=os.path.join(
                args.ckpt_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
            ),
            tokenizer_path=os.path.join(args.ckpt_dir, "xlm-roberta-large"),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device="cpu" if args.t5_cpu else self.device,
            checkpoint_path=os.path.join(
                args.ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"
            ),
            tokenizer_path=os.path.join(args.ckpt_dir, "google/umt5-xxl"),
        )
        self.audio_encoder = (
            Wav2Vec2Model.from_pretrained(
                args.wav2vec_dir, local_files_only=True, torch_dtype=torch.bfloat16
            )
            .to(self.device, dtype=torch.bfloat16)
            .eval()
        )
        self.wav2vec_fe = Wav2Vec2FeatureExtractor.from_pretrained(
            args.wav2vec_dir, local_files_only=True
        )

        torch.cuda.empty_cache()
        self.blksz_lst = [6, 8]
        self.frame_len = (self.height // (self.patch_size[1] * self.vae_stride[1])) * (
            self.width // (self.patch_size[2] * self.vae_stride[2])
        )
        kv_tokens = self.frame_len * sum(self.blksz_lst) // self.world_size
        kv_dtype = torch.float8_e4m3fn if args.fp8_kv_cache else torch.bfloat16
        kv_scale_shape = (1, kv_tokens, 40, 1)
        self.kv_cache = {
            i: {
                l: {
                    "k": torch.zeros(
                        [1, kv_tokens, 40, 128], dtype=kv_dtype, device=self.device
                    ),
                    "v": torch.zeros(
                        [1, kv_tokens, 40, 128], dtype=kv_dtype, device=self.device
                    ),
                    "k_scale": torch.ones(
                        kv_scale_shape, dtype=torch.float32, device=self.device
                    )
                    if args.fp8_kv_cache
                    else None,
                    "v_scale": torch.ones(
                        kv_scale_shape, dtype=torch.float32, device=self.device
                    )
                    if args.fp8_kv_cache
                    else None,
                    "mean_memory": False,
                    "offload_cache": False,
                    "fp8_kv_cache": args.fp8_kv_cache,
                }
                for l in range(40)
            }
            for i in range(len(self.timesteps) - 1)
        }
        for n in range(40):
            self.wan.blocks[n].self_attn.init_kvidx(self.frame_len, self.world_size)

        self.vae.model.eval()
        if not args.no_compile:
            self.vae.decode = torch.compile(self.vae.decode)

        # Swap in the tiny decoder AFTER any torch.compile of the real one.
        # self.vae.encode is untouched -- the reference image still goes through
        # the full Wan VAE, only the per-chunk decode is replaced.
        if getattr(args, "fast_decode", False):
            import sys as _sys

            _sys.path.insert(0, "/workspace/uflash")
            _sys.path.insert(0, "/workspace/uflash/UltraFlash/inference")

            _which = getattr(args, "tiny_decoder", "taew2_1")
            _lx = "/workspace/uflash/ckpt_lx2v"
            _dev = f"cuda:{self.device}"

            if _which == "v3":
                # Reconstructed from the checkpoint's shape signature; kept only
                # for comparison. MAE 6.41 vs taew2_1's 2.77.
                from ultra_dec_v3 import UltraDecoderV3 as _V3

                _d = (_V3(args.fast_decode_ckpt).to(_dev, torch.float16)
                      .eval().requires_grad_(False))

                def _fast_decode(latent, _m=_d):
                    return _m.decode_latents(latent.unsqueeze(0))

            else:
                # need_scaled is per-checkpoint, NOT a preference -- see module docs.
                _NEEDS_SCALE = {"taew2_1": False, "lighttaew2_1": True}
                if _which not in _NEEDS_SCALE:
                    raise ValueError(
                        f"--tiny_decoder must be one of "
                        f"{['v3'] + list(_NEEDS_SCALE)} (got {_which})"
                    )
                from lightx2v.models.video_encoders.hf.wan.vae_tiny import (
                    WanVAE_tiny as _Tiny,
                )

                _d = _Tiny(
                    vae_path=f"{_lx}/{_which}.pth",
                    dtype=torch.bfloat16,
                    device=_dev,
                    need_scaled=_NEEDS_SCALE[_which],
                ).to(_dev).eval()

                def _fast_decode(latent, _m=_d):
                    # (C,T,h,w) -> (1,3,T*4-3,h*8,w*8), as WanVAE.decode returns
                    return _m.decode(latent)

            self.fast_dec = _d
            self.vae.decode = _fast_decode
            if self.rank == 0:
                print(f"[fastdec] tiny decoder active: {_which}", flush=True)

        # per-chunk geometry
        self.frame_num_init = (sum(self.blksz_lst) - 1) * 4 + 1  # 53
        self.adv_frames = (
            self.blksz_lst[-1] * self.vae_stride[0]
        )  # 32 frames advanced/chunk
        self.first_frames = (
            self.blksz_lst[0] * self.vae_stride[0] - 3
        )  # 21 frames emitted by chunk 0
        self.spf = SR // self.fps if SR % self.fps == 0 else None
        assert SR % self.fps == 0, "fps must divide 16000 (use 20 or 25)"

        print("warmup...", flush=True)
        t0 = time.perf_counter()
        self._warmup()
        print(f"warmup done in {time.perf_counter() - t0:.1f}s", flush=True)

    # --- helpers -------------------------------------------------------------
    def _reset_kv(self):
        for i in self.kv_cache:
            for l in self.kv_cache[i]:
                self.kv_cache[i][l]["k"].zero_()
                self.kv_cache[i][l]["v"].zero_()

    def _encode_text(self, text):
        return [
            self.text_encoder(
                texts=text, device="cpu" if self.args.t5_cpu else self.device
            )[0].to(self.device, dtype=torch.bfloat16)
        ]

    def _embed_window(self, raw16k):
        """raw float32 @16k covering the chunk window -> audio_embs for the model."""
        wav = torch.from_numpy(raw16k).unsqueeze(0)
        rate = 25 / self.fps
        y, sr2 = torchaudio.sox_effects.apply_effects_tensor(
            wav, SR, [["tempo", f"{rate}"]]
        )
        y = T.Resample(sr2, SR)(y) * 3.0 if sr2 != SR else y * 3.0
        emb = get_embedding(
            y[0], self.wav2vec_fe, self.audio_encoder, device=self.device
        )
        return get_audio_emb(emb, 0, self.frame_num_init, self.device)

    def _warmup(self):
        # identical shapes to a real session; 2 dummy chunks
        if dist.is_initialized():
            dist.barrier()
        with torch.no_grad():
            cond = torch.randn(
                1,
                3,
                1,
                self.height,
                self.width,
                device=self.device,
                dtype=torch.bfloat16,
            ).clamp_(-1, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                clip_ctx = self.clip.visual(cond)
            ref_masks = torch.ones(
                3,
                self.height // self.vae_stride[1],
                self.width // self.vae_stride[2],
                device=self.device,
                dtype=torch.bfloat16,
            )
            pad = torch.zeros(
                1,
                3,
                self.frame_num_init - 1,
                self.height,
                self.width,
                device=self.device,
                dtype=torch.bfloat16,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = (
                    self.vae.encode(torch.concat([cond, pad], dim=2))
                    .to(self.device)
                    .unsqueeze(0)
                )
            msk = get_msk(self.frame_num_init, cond, self.vae_stride, self.device)
            y = torch.concat([msk, y], dim=1)
            ctx = self._encode_text(WORLD_PROMPT)
            dummy_win = np.zeros(
                int(SR * (self.frame_num_init + 2) / self.fps), dtype=np.float32
            )
            audio_embs = self._embed_window(dummy_win)
            pre_latent = None
            for it in range(2):
                f = 0 if it == 0 else 1
                latent = torch.randn(
                    16,
                    self.blksz_lst[f],
                    self.height // self.vae_stride[1],
                    self.width // self.vae_stride[2],
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for i in range(len(self.timesteps) - 1):
                        arg = {
                            "context": ctx,
                            "clip_fea": clip_ctx,
                            "ref_target_masks": ref_masks,
                            "audio": audio_embs,
                            "y": y[:, :, : self.frame_num_init // 4 + 1][
                                :,
                                :,
                                sum(self.blksz_lst[:f]) : sum(self.blksz_lst[: f + 1]),
                            ],
                            "start_idx": sum(self.blksz_lst[:f]) * self.frame_len,
                            "end_idx": sum(self.blksz_lst[: f + 1]) * self.frame_len,
                            "update_cache": it > 1,
                        }
                        noise = self.wan(
                            [latent],
                            t=self.timesteps[i],
                            kv_cache=self.kv_cache[i],
                            skip_audio=False if i in [1, 2] else True,
                            **arg,
                        )[0]
                        dt = (self.timesteps[i] - self.timesteps[i + 1]) / 1000
                        latent = latent + (-noise) * dt[0]
                    if it == 0:
                        _ = self.vae.decode(latent)
                    else:
                        _ = self.vae.decode(
                            torch.concat([pre_latent[:, -3:], latent], dim=1)
                        )[:, :, 9:]
                    pre_latent = latent
                torch.cuda.synchronize(self.device)
        self._reset_kv()
        if dist.is_initialized():
            dist.barrier()

    # --- the live session ----------------------------------------------------
    def run_session(self, params):
        """Runs on ALL ranks. rank0 additionally drives io/HLS."""
        img_path = params["img_path"]
        main_prompt = params.get("main_prompt") or WORLD_PROMPT

        self._reset_kv()
        image = Image.open(img_path).convert("RGB")
        cond = (
            self.transform(image)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(self.device, torch.bfloat16)
        )
        with torch.no_grad():
            clip_ctx = self.clip.visual(cond)
        # dedicated RNG: rank0-only ops (kokoro TTS, LLM sampling) consume the global
        # CUDA RNG and desync latent noise across SP ranks -> half-frame corruption
        latent_gen = torch.Generator(device=f"cuda:{self.device}")
        latent_gen.manual_seed(self.args.seed)
        ref_masks = torch.ones(
            3,
            self.height // self.vae_stride[1],
            self.width // self.vae_stride[2],
            device=self.device,
            dtype=torch.bfloat16,
        )
        msk = get_msk(self.frame_num_init, cond, self.vae_stride, self.device)
        pad = torch.zeros(
            1,
            3,
            self.frame_num_init - 1,
            self.height,
            self.width,
            device=self.device,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            y = (
                self.vae.encode(torch.concat([cond, pad], dim=2))
                .to(self.device)
                .unsqueeze(0)
            )
        y = torch.concat([msk, y], dim=1)
        y_cut = y[:, :, : self.frame_num_init // 4 + 1, ...]

        # world + event decomposition: world = persistent identity/scene, state =
        # sustained held posture (sticky), transition = transient motion (held then
        # dropped). The composed prompt is re-encoded only when it actually changes.
        world_prompt = main_prompt
        state_prompt = ""  # sustained posture/activity
        transition_prompt = None  # transient motion verb
        transition_ttl = 0

        def compose_prompt():
            parts = [world_prompt]
            if state_prompt:
                parts.append(state_prompt)
            if transition_prompt:
                parts.append(transition_prompt)
            return " ".join(parts)

        cur_prompt_str = compose_prompt()
        cur_ctx = self._encode_text(cur_prompt_str)

        # rank0 io: HLS ffmpeg (video stdin + audio fifo), writer threads
        hls_proc, vq, aq = None, None, None
        audio_hist = np.zeros(0, dtype=np.float32)
        hist_start = 0  # absolute sample index of audio_hist[0]
        drained = 0  # absolute samples pulled from speech queue timeline
        if self.rank == 0:
            live_dir = os.path.join(HLS_ROOT, "live")
            if os.path.exists(live_dir):
                shutil.rmtree(live_dir)
            os.makedirs(live_dir, exist_ok=True)
            fifo = os.path.join(live_dir, "audio.fifo")
            if os.path.exists(fifo):
                os.remove(fifo)
            os.mkfifo(fifo)
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "warning",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.out_width}x{self.out_height}",
                "-r",
                str(self.fps),
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-ar",
                str(SR),
                "-ac",
                "1",
                "-i",
                fifo,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-g",
                str(self.fps),
                "-keyint_min",
                str(self.fps),
                "-sc_threshold",
                "0",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-f",
                "hls",
                "-hls_time",
                "1",
                "-hls_list_size",
                "6",
                "-hls_segment_type",
                "mpegts",
                "-hls_flags",
                "delete_segments+append_list+independent_segments",
                os.path.join(live_dir, M3U8_NAME),
            ]
            hls_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0)
            vq, aq = queue.Queue(maxsize=8), queue.Queue(maxsize=8)

            def vworker():
                while True:
                    item = vq.get()
                    if item is None:
                        try:
                            hls_proc.stdin.close()
                        except Exception:
                            pass
                        return
                    arr, start_frame = item  # arr: [T,H,W,C] uint8
                    try:
                        hls_proc.stdin.write(arr.tobytes())
                    except Exception as e:
                        print(f"[hls video] {e}", flush=True)
                    try:
                        for i in range(arr.shape[0]):
                            buf = io.BytesIO()
                            Image.fromarray(arr[i]).save(buf, "JPEG", quality=82)
                            ws_broadcast(
                                ws_pack(0, (start_frame + i) / self.fps, buf.getvalue())
                            )
                    except Exception as e:
                        print(f"[ws video] {e}", flush=True)

            def aworker():
                try:
                    afd = open(fifo, "wb")
                except Exception as e:
                    print(f"[hls audio open] {e}", flush=True)
                    return
                while True:
                    item = aq.get()
                    if item is None:
                        try:
                            afd.close()
                        except Exception:
                            pass
                        return
                    pcm_bytes, start_sample = item
                    try:
                        afd.write(pcm_bytes)
                        afd.flush()
                    except Exception as e:
                        print(f"[hls audio] {e}", flush=True)
                        return
                    try:
                        ws_broadcast(ws_pack(1, start_sample / SR, pcm_bytes))
                    except Exception as e:
                        print(f"[ws audio] {e}", flush=True)

            threading.Thread(target=vworker, daemon=True).start()
            threading.Thread(target=aworker, daemon=True).start()
            # Clear any stop requested BEFORE this session existed: a restart
            # (/session/start while active) sets stop_flag to end the previous
            # session, and that request must not also kill this one at its first
            # chunk boundary if the previous session had already exited.
            STATE.stop_flag = False
            STATE.active = True
            STATE.chunk_count = 0
            STATE.log("system", "session started")

        def drain_to(n_abs):
            """rank0: make the speech timeline defined up to absolute sample n_abs."""
            nonlocal audio_hist, hist_start, drained
            while drained < n_abs:
                try:
                    piece = STATE.speech_q.get_nowait()
                except queue.Empty:
                    piece = np.zeros(min(n_abs - drained, SR // 5), dtype=np.float32)
                audio_hist = np.concatenate([audio_hist, piece])
                drained += len(piece)
            # keep last ~8s
            max_keep = SR * 8
            if len(audio_hist) > max_keep:
                cut = len(audio_hist) - max_keep
                audio_hist = audio_hist[cut:]
                hist_start += cut

        def slice_abs(a, b):
            """absolute sample range -> float32 array (zeros where undefined)."""
            out = np.zeros(b - a, dtype=np.float32)
            lo, hi = max(a, hist_start), min(b, hist_start + len(audio_hist))
            if hi > lo:
                out[lo - a : hi - a] = audio_hist[lo - hist_start : hi - hist_start]
            return out

        pre_latent = None
        sr_u8 = None
        sr_job = None
        sr_inflight = []
        if self.sr_url and self.rank == 0:
            _sr_reset(self.sr_url)
        iteration = 0
        wall_start = None
        frames_emitted = 0
        audio_samples_emitted = 0
        try:
            while True:
                # ---- chunk-boundary control sync ----
                if self.rank == 0:
                    stop = STATE.stop_flag
                    directive = None
                    with STATE.lock:
                        if STATE.pending_directive is not None:
                            directive = STATE.pending_directive
                            STATE.pending_directive = None
                    # window in frames: [start_f, end_f+2)
                    if iteration == 0:
                        start_f, end_f = 0, self.frame_num_init
                    else:
                        start_f = (iteration - 1) * self.adv_frames
                        end_f = start_f + self.frame_num_init
                    need = (end_f + 2) * self.spf
                    drain_to(need)
                    win = slice_abs(start_f * self.spf, (end_f + 2) * self.spf)
                    # heard audio = frames advanced by this chunk
                    if iteration == 0:
                        heard = slice_abs(0, self.first_frames * self.spf)
                    else:
                        a0 = (
                            self.first_frames + (iteration - 1) * self.adv_frames
                        ) * self.spf
                        heard = slice_abs(a0, a0 + self.adv_frames * self.spf)
                    payload = {"stop": stop, "directive": directive, "win": win}
                else:
                    payload = None
                if self.use_dist:
                    box = [payload]
                    dist.broadcast_object_list(box, src=0, group=self.control_pg)
                    payload = box[0]
                if payload["stop"]:
                    break
                if payload["directive"] is not None:
                    d = payload["directive"]
                    rest = d.get("rest")
                    # rest None -> gesture (leave sustained state); str sets it, "" clears
                    if rest is not None:
                        state_prompt = rest.strip()
                    trans = (d.get("transition") or "").strip()
                    if trans:
                        transition_prompt = trans
                        transition_ttl = self.args.action_hold
                    else:
                        transition_prompt = None
                        transition_ttl = 0

                # re-encode only when the composed world+state+transition changes
                tgt = compose_prompt()
                if tgt != cur_prompt_str:
                    cur_ctx = self._encode_text(tgt)
                    cur_prompt_str = tgt

                audio_embs = self._embed_window(payload["win"])

                f = 0 if iteration == 0 else 1
                latent = torch.randn(
                    16,
                    self.blksz_lst[f],
                    self.height // self.vae_stride[1],
                    self.width // self.vae_stride[2],
                    dtype=torch.bfloat16,
                    device=f"cuda:{self.device}",
                    generator=latent_gen,
                )
                t0 = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for i in range(len(self.timesteps) - 1):
                        arg = {
                            "context": cur_ctx,
                            "clip_fea": clip_ctx,
                            "ref_target_masks": ref_masks,
                            "audio": audio_embs,
                            "y": y_cut[
                                :,
                                :,
                                sum(self.blksz_lst[:f]) : sum(self.blksz_lst[: f + 1]),
                            ],
                            "start_idx": sum(self.blksz_lst[:f]) * self.frame_len,
                            "end_idx": sum(self.blksz_lst[: f + 1]) * self.frame_len,
                            "update_cache": iteration > 1,
                        }
                        noise = self.wan(
                            [latent],
                            t=self.timesteps[i],
                            kv_cache=self.kv_cache[i],
                            skip_audio=False if i in [1, 2] else True,
                            **arg,
                        )[0]
                        dt = (self.timesteps[i] - self.timesteps[i + 1]) / 1000
                        latent = latent + (-noise) * dt[0]
                    if self.sr_url:
                        # Sidecar owns upsampler + SR DiT + tiny decoder and keeps
                        # its own streaming caches, so nothing is decoded here.
                        vids = None
                        sr_u8 = None
                        sr_job = (
                            (
                                latent.detach().to(torch.bfloat16).cpu(),
                                [c.detach().to(torch.bfloat16).cpu() for c in cur_ctx],
                            )
                            if self.rank == 0
                            else None
                        )
                    elif iteration == 0:
                        vids = self.vae.decode(latent)
                    else:
                        vids = self.vae.decode(
                            torch.concat([pre_latent[:, -3:], latent], dim=1)
                        )[:, :, 9:]
                    pre_latent = latent

                # SOULX_DUMP_LATENTS: tap the chunk latent + its T5 context on the
                # way to the VAE. This is where a latent-space SR stage would hook in.
                if _DUMP_DIR and self.rank == 0 and iteration < _DUMP_N:
                    import os as _os

                    _os.makedirs(_DUMP_DIR, exist_ok=True)
                    torch.save(
                        {
                            "latent": latent.detach().to(torch.bfloat16).cpu(),
                            "ctx": [c.detach().to(torch.bfloat16).cpu() for c in cur_ctx],
                            "iteration": iteration,
                            "size": (self.width, self.height),
                            "fps": self.fps,
                            "prompt": cur_prompt_str,
                        },
                        f"{_DUMP_DIR}/chunk_{iteration:04d}.pt",
                    )
                    print(f"[dump] chunk {iteration} -> {_DUMP_DIR}", flush=True)

                n_emit = 0
                if self.rank == 0 and self.sr_url:
                    # submit chunk N, emit chunk N-1 -> SR overlaps the next denoise
                    sr_inflight.append(
                        (_sr_pool.submit(_sr_request, self.sr_url,
                                         sr_job[0], sr_job[1], 0), heard)
                    )
                    if len(sr_inflight) >= 2:
                        fut0, heard = sr_inflight.pop(0)
                        sr_u8 = fut0.result()
                    else:
                        sr_u8 = None

                if self.rank == 0 and (sr_u8 is not None or not self.sr_url):
                    if sr_u8 is not None:
                        arr = sr_u8
                    else:
                        arr = (
                            ((vids.squeeze(0).permute(1, 2, 3, 0) + 1.0) * 127.5)
                            .clamp(0, 255)
                            .to(torch.uint8)
                            .contiguous()
                            .cpu()
                            .numpy()
                        )
                    n_emit = arr.shape[0]
                    vq.put((arr, frames_emitted))
                    aq.put(
                        (
                            (np.clip(heard, -1, 1) * 32767).astype(np.int16).tobytes(),
                            audio_samples_emitted,
                        )
                    )
                    audio_samples_emitted += len(heard)
                    STATE.chunk_count = iteration + 1
                    if iteration % 10 == 0:
                        dt_s = time.perf_counter() - t0
                        n_frames = arr.shape[0]
                        mem_gb = torch.cuda.memory_allocated(self.device) / 1e9
                        print(
                            f"chunk {iteration}: {n_frames}f in {dt_s:.2f}s "
                            f"({n_frames / dt_s:.1f} fps) mem={mem_gb:.1f}GB "
                            f"state={state_prompt or 'idle'!r} trans={transition_prompt!r}",
                            flush=True,
                        )

                # pace to wall clock (keep ~1 chunk of lead) so commands stay realtime
                if self.rank == 0:
                    if wall_start is None:
                        wall_start = time.perf_counter()
                    frames_emitted += n_emit
                    target = wall_start + frames_emitted / self.fps - 1.6
                    lag = target - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)

                # transition expires -> drop the transient motion; the sustained state
                # prompt + rolling KV cache keep the new pose, so she stays (e.g.)
                # seated instead of snapping back to idle. Re-encode happens at the top
                # of the next chunk via compose_prompt().
                if transition_prompt is not None:
                    transition_ttl -= 1
                    if transition_ttl <= 0:
                        transition_prompt = None
                iteration += 1
        finally:
            if self.rank == 0:
                try:
                    vq.put(None)
                    aq.put(None)
                except Exception:
                    pass
                if hls_proc is not None:
                    try:
                        hls_proc.wait(timeout=10)
                    except Exception:
                        hls_proc.kill()
                STATE.active = False
                STATE.stop_flag = False
                STATE.log("system", "session stopped")
            torch.cuda.empty_cache()
            gc.collect()


# ----------------------------------------------------------------------------
# control loops
# ----------------------------------------------------------------------------
def control_loop_rank0():
    consecutive_failures = 0
    while True:
        try:
            # 15s heartbeat: rank1 blocks on this broadcast while idle and gloo
            # recv times out (30min default) without periodic traffic
            params = STATE.start_params.get(timeout=15)
        except queue.Empty:
            params = None
        if engine.use_dist:
            box = [params]
            dist.broadcast_object_list(box, src=0, group=engine.control_pg)
        if params is None:
            continue
        try:
            engine.run_session(params)
            consecutive_failures = 0
        except Exception as e:
            import traceback

            traceback.print_exc()
            STATE.last_error = str(e)
            STATE.active = False
            STATE.stop_flag = False
            STATE.log("error", str(e))
            consecutive_failures += 1
            if consecutive_failures <= 3:
                STATE.log(
                    "system",
                    f"auto-restarting session (attempt {consecutive_failures}/3)",
                )
                time.sleep(5)
                STATE.start_params.put(params)


def control_loop_other():
    while True:
        box = [None]
        dist.broadcast_object_list(box, src=0, group=engine.control_pg)
        params = box[0]
        if params is None:
            continue
        try:
            engine.run_session(params)
        except Exception:
            import traceback

            traceback.print_exc()


# ----------------------------------------------------------------------------
# Flask routes (rank0)
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "chat.html",
        stream_resolution=f"{engine.width}x{engine.height}",
        stream_w=engine.width,
        stream_h=engine.height,
    )


@app.route("/character/default")
def default_character():
    return send_file(engine.args.image)


@app.route("/session/start", methods=["POST"])
def session_start():
    img_path = engine.args.image
    f = request.files.get("img_file")
    if f:
        # unique name per upload: the previous reference may still be mmap'd by
        # the session we are about to replace, and overwriting it in place also
        # lets a stale browser cache serve the old picture back
        img_path = os.path.join(
            BASE_DIR, "uploads", f"session_input_{time.time_ns()}.png"
        )
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        f.save(img_path)
        try:
            with Image.open(img_path) as uploaded:
                uploaded.verify()
        except Exception:
            os.remove(img_path)
            return jsonify({"status": "error", "message": "invalid image"}), 400
        _prune_uploads(os.path.dirname(img_path), keep=5)
    main_prompt = (request.form.get("main_prompt") or "").strip() or WORLD_PROMPT
    configure_persona(request.form)
    brain.reset_history()
    # Restart ONLY when the video inputs actually change, i.e. a new reference
    # image. A start with no upload on a live session changes nothing on screen,
    # and tearing the session down would drop the KV cache and freeze the stream
    # for seconds. The client cannot be trusted to know: its `active` flag is
    # false until the first /status poll, so every page load raced into this.
    if STATE.active and not f:
        STATE.log("system", "persona updated (session kept)")
        return jsonify(
            {"status": "ok", "restarting": False, "persona": persona_payload()}
        )
    with STATE.speech_q.mutex:
        STATE.speech_q.queue.clear()
    # The reference image is encoded once at session start and then lives in the
    # rolling KV cache, so a new one means a NEW session. Rather than making the
    # client stop/poll/start, queue the params and ask any running session to
    # stop: control_loop_rank0 is blocked inside run_session and only pops
    # start_params once that returns, so the swap happens in order.
    restarting = STATE.active
    with STATE.lock:
        while True:  # drop superseded requests so we start exactly one session
            try:
                STATE.start_params.get_nowait()
            except queue.Empty:
                break
        STATE.start_params.put({"img_path": img_path, "main_prompt": main_prompt})
        STATE.stop_flag = restarting
    STATE.log("system", "restarting with a new character" if restarting else "starting")
    return jsonify(
        {"status": "ok", "restarting": restarting, "persona": persona_payload()}
    )


@app.route("/session/stop", methods=["POST"])
def session_stop():
    STATE.stop_flag = True
    return jsonify({"status": "ok"})


def configure_persona(values):
    name = (values.get("name") or "").strip()[:40]
    persona = (values.get("persona") or "").strip()[:500]
    voice = (values.get("voice") or "").strip()
    with STATE.lock:
        if name:
            STATE.persona_name = name
        if persona:
            STATE.persona_prompt = persona
        if voice in KOKORO_VOICES:
            STATE.voice = voice


def persona_payload():
    with STATE.lock:
        return {
            "name": STATE.persona_name,
            "persona": STATE.persona_prompt,
            "voice": STATE.voice,
        }


@app.route("/persona", methods=["POST"])
def persona():
    configure_persona(request.get_json(force=True))
    brain.reset_history()
    STATE.log("system", f"persona updated: {STATE.persona_name}")
    return jsonify({"status": "ok", "persona": persona_payload()})


@app.route("/say", methods=["POST"])
def say():
    text = (request.get_json(force=True).get("text") or "").strip()
    if not text:
        return jsonify({"status": "error", "message": "empty"}), 400
    try:
        wav = tts.synth(text, voice=STATE.voice)
    except Exception as e:
        STATE.log("error", f"TTS failed: {e}")
        return jsonify({"status": "error", "message": f"TTS failed: {e}"}), 500
    if len(wav):
        STATE.speech_q.put(wav)
    STATE.log("say", text)
    return jsonify({"status": "ok", "seconds": round(len(wav) / SR, 2)})


@app.route("/action", methods=["POST"])
def action():
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    pose = body.get("pose")  # explicit resting state; None -> gesture, "" -> clear
    persist = bool(body.get("persist"))
    if not text and pose is None and not persist:
        # explicit idle: drop any transition and clear the sustained state
        directive = {"transition": None, "rest": ""}
    else:
        rest = pose if pose is not None else (text if persist else None)
        directive = {"transition": text or None, "rest": rest}
    with STATE.lock:
        STATE.pending_directive = directive
    STATE.log("action", text or "(idle)")
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    msg = (request.get_json(force=True).get("text") or "").strip()
    if not msg:
        return jsonify({"status": "error", "message": "empty"}), 400
    STATE.log("user", msg)
    try:
        say_text, action_text, pose_text = brain.reply(msg)
    except Exception as e:
        STATE.log("error", f"LLM failed: {e}")
        return jsonify({"status": "error", "message": f"LLM failed: {e}"}), 500
    # action = transient motion; pose = sustained state to keep afterwards (None -> gesture)
    if action_text or pose_text:
        with STATE.lock:
            STATE.pending_directive = {"transition": action_text, "rest": pose_text}
        STATE.log("action", action_text or f"(pose: {pose_text})")
    if say_text:
        try:
            wav = tts.synth(say_text, voice=STATE.voice)
            if len(wav):
                STATE.speech_q.put(wav)
        except Exception as e:
            STATE.log("error", f"TTS failed: {e}")
        STATE.log("chano", say_text)
    return jsonify(
        {"status": "ok", "say": say_text, "action": action_text, "pose": pose_text}
    )


@app.route("/status")
def status():
    with STATE.lock:
        ev = list(STATE.events)
    return jsonify(
        {
            "active": STATE.active,
            "chunks": STATE.chunk_count,
            "queued_speech_s": round(
                sum(len(x) for x in list(STATE.speech_q.queue)) / SR, 1
            ),
            "error": STATE.last_error,
            "events": ev,
            "persona": persona_payload(),
        }
    )


@app.route("/stream/live/<path:filename>")
def serve_hls(filename):
    return send_from_directory(os.path.join(HLS_ROOT, "live"), filename)


@sock.route("/ws")
def ws_feed(ws):
    q = queue.Queue(maxsize=256)
    with WS_LOCK:
        WS_CLIENTS.add(q)
    try:
        ws.send(
            json.dumps(
                {
                    "kind": "meta",
                    "fps": engine.fps,
                    "sr": SR,
                    "w": engine.out_width,
                    "h": engine.out_height,
                }
            )
        )
        while True:
            msg = q.get()
            ws.send(msg)
    except Exception:
        pass
    finally:
        with WS_LOCK:
            WS_CLIENTS.discard(q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--wav2vec_dir", type=str, required=True)
    parser.add_argument(
        "--image", type=str, required=True, help="default reference image"
    )
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--aux_device",
        type=str,
        default="cpu",
        help="device for rank0-only LLM/TTS; CPU avoids interfering with NCCL",
    )
    parser.add_argument(
        "--aux_url",
        type=str,
        default=None,
        help="optional separate Qwen/Kokoro service URL",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=str, default="720*416")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--fp8_kv_cache", action="store_true", default=False)
    parser.add_argument("--fast_decode", action="store_true",
                        help="decode with the v3 tiny decoder instead of the Wan2.1 VAE "
                             "(~31ms vs ~452ms at 720x416, MAE 6.4/255)")
    parser.add_argument("--tiny_decoder", type=str, default="taew2_1",
                        choices=["taew2_1", "lighttaew2_1", "v3"],
                        help="which tiny decoder --fast_decode uses "
                             "(taew2_1: official, MAE 2.77; v3: reconstructed, 6.41)")
    parser.add_argument("--fast_decode_ckpt", type=str,
                        default="/workspace/uflash/ckpt/v1.1-ultra-decoder-v3-ema_decoder.pth")
    parser.add_argument("--sr_url", type=str, default=None,
                        help="UltraFlash SR sidecar base URL; omit to stream at native res")
    parser.add_argument("--sr_scale", type=int, default=2)
    parser.add_argument(
        "--action_hold",
        type=int,
        default=4,
        help="chunks to hold an action before reverting to idle prompt",
    )
    parser.add_argument(
        "--no_fp8_gemm",
        action="store_true",
        help="disable vllm FP8 GEMM entirely",
    )
    parser.add_argument(
        "--fp8_scope",
        choices=["blocks", "all"],
        default="blocks",
        help="which linears FP8 covers: 'blocks' = attention/FFN matmuls only "
        "(leaves modulation/embeddings/head in bf16), 'all' = every linear",
    )
    parser.add_argument(
        "--no_compile", action="store_true", help="disable torch.compile"
    )
    parser.add_argument(
        "--compile_blocks",
        action="store_true",
        help="compile each Wan transformer block independently (Pruna Wan recipe)",
    )
    parser.add_argument(
        "--compile_mode",
        choices=["default", "reduce-overhead", "max-autotune-no-cudagraphs"],
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument(
        "--autostart", action="store_true", help="start session immediately"
    )
    args = parser.parse_args()

    engine = LiveEngine(args)
    if engine.rank == 0:
        tts = TTS(args.aux_device, args.aux_url)
        brain = Brain(args.aux_device, args.aux_url)
        threading.Thread(target=control_loop_rank0, daemon=True).start()
        if args.autostart:
            STATE.start_params.put(
                {"img_path": args.image, "main_prompt": WORLD_PROMPT}
            )
        print(
            f"\n LiveAct interactive demo on http://0.0.0.0:{args.port}\n", flush=True
        )
        app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)
    else:
        control_loop_other()
    if dist.is_initialized():
        dist.destroy_process_group()
