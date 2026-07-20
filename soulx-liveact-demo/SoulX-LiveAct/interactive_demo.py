"""Interactive persona demo for SoulX-LiveAct.

A continuous live session driven by chat:
  - /chat    -> LLM decides {say, action}; speech goes through kokoro TTS into the
               audio buffer (lip-synced), action swaps the T5 prompt context mid-stream.
  - /say     -> speak exactly this text.
  - /action  -> perform this motion prompt now (held for a few chunks, then reverts).
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

IDLE_PROMPT = (
    "A beautiful blonde anime girl looks at the camera with a gentle smile, "
    "breathing softly, subtle natural idle motion."
)
DEFAULT_PERSONA = "Cheerful, warm, playful, curious, and wholesome."
KOKORO_VOICES = {"af_heart", "af_bella", "af_nicole", "am_adam"}


# ----------------------------------------------------------------------------
# rank0 session state (fed by Flask routes, consumed at chunk boundaries)
# ----------------------------------------------------------------------------
class SessionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.speech_q = queue.Queue()  # np.float32 arrays @16k queued for playback
        self.pending_action = None  # str | None, picked up at next chunk boundary
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
            'Reply ONLY with JSON: {"say": "<what you say>", "action": "<short third-person '
            "visual description of your motion, e.g. 'She waves her hand at the camera cheerfully!' "
            'or null if no special motion>"}. Keep everything friendly and wholesome.'
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
                    return f"You said: {message}", None
                text = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                ids = self.tok(text, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **ids,
                        max_new_tokens=160,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tok.eos_token_id,
                    )
                raw = self.tok.decode(
                    out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
                )
            say, action = raw.strip(), None
            try:
                s = raw[raw.index("{") : raw.rindex("}") + 1]
                obj = json.loads(s)
                say = obj.get("say") or ""
                action = obj.get("action") or None
                if isinstance(action, str) and action.lower() in ("null", "none", ""):
                    action = None
            except Exception:
                pass
            self.history.append({"role": "assistant", "content": say})
            return say, action


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
        self.width, self.height = [int(x) for x in args.size.split("*")]
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
            enable_fp8_gemm(self.wan, options=FP8GemmOptions())
        self.wan = self.wan.to(self.device)
        self.wan.freqs = self.wan.freqs.to(self.device)
        self.wan.eval()
        if not args.no_compile:
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
            ctx = self._encode_text(IDLE_PROMPT)
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
        main_prompt = params.get("main_prompt") or IDLE_PROMPT

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

        idle_ctx = self._encode_text(main_prompt)
        cur_ctx, cur_action, action_ttl = idle_ctx, None, 0

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
                f"{self.width}x{self.height}",
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
        iteration = 0
        wall_start = None
        frames_emitted = 0
        audio_samples_emitted = 0
        try:
            while True:
                # ---- chunk-boundary control sync ----
                if self.rank == 0:
                    stop = STATE.stop_flag
                    action = None
                    with STATE.lock:
                        if STATE.pending_action is not None:
                            action = STATE.pending_action
                            STATE.pending_action = None
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
                    payload = {"stop": stop, "action": action, "win": win}
                else:
                    payload = None
                if self.use_dist:
                    box = [payload]
                    dist.broadcast_object_list(box, src=0, group=self.control_pg)
                    payload = box[0]
                if payload["stop"]:
                    break
                if payload["action"] is not None:
                    txt = payload["action"].strip()
                    if txt:
                        cur_ctx = self._encode_text(txt)
                        cur_action, action_ttl = txt, self.args.action_hold
                    else:
                        cur_ctx, cur_action, action_ttl = idle_ctx, None, 0

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
                    if iteration == 0:
                        vids = self.vae.decode(latent)
                    else:
                        vids = self.vae.decode(
                            torch.concat([pre_latent[:, -3:], latent], dim=1)
                        )[:, :, 9:]
                    pre_latent = latent

                if self.rank == 0:
                    u8 = (
                        ((vids.squeeze(0).permute(1, 2, 3, 0) + 1.0) * 127.5)
                        .clamp(0, 255)
                        .to(torch.uint8)
                        .contiguous()
                        .cpu()
                    )
                    vq.put((u8.numpy(), frames_emitted))
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
                        n_frames = vids.shape[2]
                        mem_gb = torch.cuda.memory_allocated(self.device) / 1e9
                        print(
                            f"chunk {iteration}: {n_frames}f in {dt_s:.2f}s "
                            f"({n_frames / dt_s:.1f} fps) mem={mem_gb:.1f}GB "
                            f"action={cur_action}",
                            flush=True,
                        )

                # pace to wall clock (keep ~1 chunk of lead) so commands stay realtime
                if self.rank == 0:
                    if wall_start is None:
                        wall_start = time.perf_counter()
                    frames_emitted += vids.shape[2]
                    target = wall_start + frames_emitted / self.fps - 1.6
                    lag = target - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)

                # revert action after hold expires
                if cur_action is not None:
                    action_ttl -= 1
                    if action_ttl <= 0:
                        cur_ctx, cur_action = idle_ctx, None
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
        "chat.html", stream_resolution=engine.args.size.replace("*", "x")
    )


@app.route("/character/default")
def default_character():
    return send_file(engine.args.image)


@app.route("/session/start", methods=["POST"])
def session_start():
    if STATE.active:
        return jsonify({"status": "error", "message": "session already active"}), 429
    img_path = engine.args.image
    f = request.files.get("img_file")
    if f:
        img_path = os.path.join(BASE_DIR, "uploads", "session_input.png")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        f.save(img_path)
        try:
            with Image.open(img_path) as uploaded:
                uploaded.verify()
        except Exception:
            return jsonify({"status": "error", "message": "invalid image"}), 400
    main_prompt = (request.form.get("main_prompt") or "").strip() or IDLE_PROMPT
    configure_persona(request.form)
    STATE.stop_flag = False
    with STATE.speech_q.mutex:
        STATE.speech_q.queue.clear()
    brain.reset_history()
    STATE.start_params.put({"img_path": img_path, "main_prompt": main_prompt})
    return jsonify({"status": "ok", "persona": persona_payload()})


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
    text = (request.get_json(force=True).get("text") or "").strip()
    with STATE.lock:
        STATE.pending_action = text  # empty string -> revert to idle
    STATE.log("action", text or "(revert to idle)")
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    msg = (request.get_json(force=True).get("text") or "").strip()
    if not msg:
        return jsonify({"status": "error", "message": "empty"}), 400
    STATE.log("user", msg)
    try:
        say_text, action_text = brain.reply(msg)
    except Exception as e:
        STATE.log("error", f"LLM failed: {e}")
        return jsonify({"status": "error", "message": f"LLM failed: {e}"}), 500
    if action_text:
        with STATE.lock:
            STATE.pending_action = action_text
        STATE.log("action", action_text)
    if say_text:
        try:
            wav = tts.synth(say_text, voice=STATE.voice)
            if len(wav):
                STATE.speech_q.put(wav)
        except Exception as e:
            STATE.log("error", f"TTS failed: {e}")
        STATE.log("chano", say_text)
    return jsonify({"status": "ok", "say": say_text, "action": action_text})


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
                    "w": engine.width,
                    "h": engine.height,
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
    parser.add_argument(
        "--action_hold",
        type=int,
        default=4,
        help="chunks to hold an action before reverting to idle prompt",
    )
    parser.add_argument(
        "--no_fp8_gemm",
        action="store_true",
        help="disable vllm FP8 GEMM (produces noise on this pod)",
    )
    parser.add_argument(
        "--no_compile", action="store_true", help="disable torch.compile"
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
            STATE.start_params.put({"img_path": args.image, "main_prompt": IDLE_PROMPT})
        print(
            f"\n LiveAct interactive demo on http://0.0.0.0:{args.port}\n", flush=True
        )
        app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)
    else:
        control_loop_other()
    if dist.is_initialized():
        dist.destroy_process_group()
