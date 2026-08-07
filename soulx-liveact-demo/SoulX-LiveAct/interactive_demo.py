"""Interactive persona demo for SoulX-LiveAct.

A continuous live session driven by chat, modeled on Wan-Streamer v0.3's
"world + event stream": the T5 context is composed as world (persistent
identity/scene) + sustained state (held posture) + a transient transition.
Transitions are held a few chunks then dropped, but posture changes update the
sustained state so the character STAYS in the new pose (sit -> keeps sitting).

  - /chat    -> the LLM answers in prose with its own motions written inline,
               "Sure! [she turns around, 2s] Ta-da!"; `directions.py` splits that
               into speech -> kokoro TTS (lip-synced) and a motion -> generator,
               with the stated duration as the transition's TTL. A posture change
               with no duration becomes the new sustained state.
  - /say     -> speak exactly this text.
  - /action  -> perform a motion now; {pose} or {persist:true} makes it stick,
               empty body clears back to neutral idle.
Video+audio are muxed into a live HLS stream at /stream/live/live.m3u8.

Prefer the launcher, which picks the GPU topology and gates FP8/FA3 on the
architecture instead of assuming an H200:

  scope-soulx run --res 416x720 --vae taew2_1

Directly, if you are debugging one knob (paths come from SOULX_ROOT, see
soulx_runtime.Paths):

  python interactive_demo.py --size 416x720 --vae wan --compile off --autostart

Under sequence parallelism it is one process per rank:

  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 interactive_demo.py ...
"""

import argparse
import concurrent.futures as _cf
import gc
import io
import json
import os
import queue
import shutil
import struct
import subprocess
import threading
import time
import urllib.request
from datetime import timedelta

import directions as directions_mod
import lora as lora_mod
import numpy as np
import soulx_runtime
import torch
import torch.distributed as dist
import torchaudio
import torchaudio.transforms as T
from appearance import (
    AppearanceError,
    AppearanceRateLimit,
    FalAppearanceEditor,
    parse_change_command,
)
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
from flask_sock import Sock
from fp8_gemm import FP8GemmOptions, enable_fp8_gemm
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
from PIL import Image
from torchvision import transforms
from transformers import Wav2Vec2FeatureExtractor
from util_liveact import (
    center_rescale_crop_keep_ratio,
    get_audio_emb,
    get_embedding,
    get_msk,
)
from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel

from src.audio_analysis.wav2vec2 import Wav2Vec2Model

gc.collect()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

app = Flask(__name__)
# Jinja caches the compiled template, and reloading this process costs minutes
# of weight load plus compile warmup. A stat() per page render is nothing next
# to that -- the page is fetched once per viewer, not once per frame.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
sock = Sock(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HLS_ROOT = os.environ.get("SOULX_HLS_ROOT", os.path.join(BASE_DIR, "hls_output"))
M3U8_NAME = "live.m3u8"
os.makedirs(HLS_ROOT, exist_ok=True)
appearance_editor = FalAppearanceEditor(os.path.join(BASE_DIR, "uploads"))

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
    """'416x720' / '416*720' / a preset name -> (width, height), validated.

    Portrait and landscape both allowed, and portrait costs the same: latency
    tracks pixel count, not orientation.
    """
    try:
        return soulx_runtime.parse_resolution(size)
    except ValueError as exc:
        raise SystemExit(
            f"--size {size!r}: {exc}\n  presets: {', '.join(soulx_runtime.RESOLUTIONS)}"
        ) from None


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


def _close_media_queue(q):
    """Request worker shutdown without deadlocking on a full bounded queue."""

    if q is None:
        return
    try:
        q.put_nowait(None)
    except queue.Full:
        # Drop one stale media chunk. If the writer itself is blocked, the main
        # session thread must still be able to kill ffmpeg and restart.
        try:
            q.get_nowait()
            q.put_nowait(None)
        except (queue.Empty, queue.Full):
            pass


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
        # how many chunks a transient motion survives before it is dropped and
        # the sustained state takes back over. Seeded from --action_hold at
        # startup and mutable at runtime via POST /action_hold, because the
        # right value is a feel judgement and a restart costs a ~3min warmup.
        self.action_hold = 8
        # what the generator is actually holding right now, mirrored out of the
        # chunk loop for /status. Read-only from the Flask side: the loop owns
        # these, and until now the only way to see the sticky state was to grep
        # the chunk logs.
        self.state_prompt = ""
        self.transition_prompt = None
        self.transition_ttl = 0
        # newest emitted frame, so /anchor can make the pose she is CURRENTLY in
        # the new reference image. Plain attribute: one writer (the video worker)
        # and readers only ever want the latest, so a lock would buy nothing.
        self.last_frame = None
        # new LoRA strength, applied at the next chunk boundary (never mid-
        # forward: the merge rewrites bf16 masters and re-quantizes fp8).
        self.pending_lora_strength = None
        self.stop_flag = False
        self.start_params = queue.Queue()  # session start requests
        self.active = False
        self.chunk_count = 0
        self.last_error = None
        self.events = []  # rolling log for the UI
        self.persona_name = "Chano"
        self.persona_prompt = DEFAULT_PERSONA
        self.voice = "af_heart"
        self.current_image_path = None
        self.current_main_prompt = WORLD_PROMPT
        self.image_revision = 0
        self.appearance_busy = False

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

# WebRTC transport, opt-in via --webrtc. Runs ALONGSIDE the WebSocket path off
# the same emit worker, so one session can serve both and be compared directly.
# aiortc is an optional dependency: without it, --webrtc reports why and the
# demo runs exactly as before.
WEBRTC = None
soulx_webrtc = None


def _enable_webrtc(fps):
    """Bring up the WebRTC hub, or explain why it stayed off.

    Imported here rather than at module scope so aiortc stays a genuinely
    optional dependency -- a box without it runs the WebSocket path unchanged
    instead of failing at startup.
    """
    global WEBRTC, soulx_webrtc
    try:
        import soulx_webrtc as _mod
    except ImportError as e:
        print(f"[webrtc] disabled: {e} -- pip install aiortc", flush=True)
        return
    soulx_webrtc = _mod
    WEBRTC = _mod.WebRTCHub(fps)
    ice = _mod.ice_servers()
    where = ice[0]["urls"] if ice else "no TURN (set SOULX_TURN_URL); direct UDP only"
    print(f"[webrtc] enabled, H.264/Opus; ICE: {where}", flush=True)


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
        # Prompt and parser ship together in `directions.py` — see the note there.
        return directions_mod.system_prompt(STATE.persona_name, STATE.persona_prompt)

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
        """-> directions.Direction (say, action, pose, hold)"""
        with self.lock:
            self.history.append({"role": "user", "content": message})
            msgs = (
                [{"role": "system", "content": self._system_prompt()}]
                + directions_mod.FEWSHOT
                + self.history[-12:]
            )
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
                    return directions_mod.Direction(say=f"You said: {message}")
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
            direction = directions_mod.parse(
                raw, name=STATE.persona_name, default_hold=STATE.action_hold
            )
            # The script, not the bare speech: the model's own bracketed line is
            # the strongest example of the format it sees next turn.
            self.history.append({"role": "assistant", "content": direction.script})
            return direction


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

        # Resolve what this card can actually do BEFORE importing the model: the
        # attention backend is chosen at model_memory import time, and FP8 on a
        # card without FP8 tensor cores fails deep inside the first matmul rather
        # than at startup. The launcher plans GPU topology; this only gates the
        # per-architecture features, so a bare `python interactive_demo.py` is
        # still safe on any box.
        os.environ["SOULX_ATTN"] = soulx_runtime.attention_backend(args.attn)
        os.environ["SOULX_SAGE_SCOPE"] = args.sage_scope
        if args.fp8 != "off" and not soulx_runtime.fp8_supported():
            print(
                f"[plan] fp8={args.fp8} requested but this GPU has no FP8 tensor "
                f"cores (needs sm_89+); falling back to bf16",
                flush=True,
            )
            args.fp8 = "off"
        if self.rank == 0:
            for gpu in soulx_runtime.gpus():
                print(f"[gpu] {gpu}", flush=True)
            print(
                f"[plan] size={self.width}x{self.height} vae={args.vae} "
                f"attn={os.environ['SOULX_ATTN']} fp8={args.fp8} "
                f"compile={args.compile} world_size={self.world_size}",
                flush=True,
            )

        if self.use_dist:
            from model_liveact.model_memory_sp import WanModel
        else:
            from model_liveact.model_memory import WanModel
        self.wan = WanModel.from_pretrained(
            args.ckpt_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
        ).to(dtype=torch.bfloat16)
        # LoRA goes in here, on CPU, while the targets are still plain
        # nn.Linear: after enable_fp8_gemm the bf16 weights are quantized (and
        # by default freed), and after torch.compile a weight edit invalidates
        # every graph. See lora.py for the merge-vs-adapter reasoning.
        self.lora = lora_mod.load_and_merge(self.wan, args.lora, args.lora_strength)
        if args.fp8 != "off":
            mf = None if args.fp8 == "all" else _fp8_block_matmuls
            # Keeping the bf16 masters on CPU is what lets /lora change the
            # strength without a restart. ~28GB of host RAM on the wrapped
            # linears, no VRAM cost -- the weights leave the GPU either way.
            storage = "cpu_offload" if self.lora is not None else "discard"
            enable_fp8_gemm(
                self.wan,
                options=FP8GemmOptions(fp16_weight_storage=storage),
                module_filter=mf,
            )
            wrapped = sum(
                1 for m in self.wan.modules() if type(m).__name__ == "FP8Linear"
            )
            print(
                f"fp8 gemm: scope={args.fp8}, {wrapped} linears wrapped",
                flush=True,
            )
        self.wan = self.wan.to(self.device)
        self.wan.freqs = self.wan.freqs.to(self.device)
        self.wan.eval()
        if args.compile == "blocks":
            # Pruna's open-source Wan recipe: compile the transformer ModuleList
            # one block at a time. More robust to graph breaks than compiling the
            # stateful streaming WanModel as one graph, and worth 17-20%.
            for i, block in enumerate(self.wan.blocks):
                self.wan.blocks[i] = torch.compile(
                    block, mode=args.compile_mode, backend="inductor", dynamic=False
                )
            print(
                f"torch.compile: {len(self.wan.blocks)} blocks, mode={args.compile_mode}",
                flush=True,
            )
        elif args.compile == "full":
            self.wan = torch.compile(
                self.wan,
                mode=args.compile_mode,
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
                layer: {
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
                for layer in range(40)
            }
            for i in range(len(self.timesteps) - 1)
        }
        for n in range(40):
            self.wan.blocks[n].self_attn.init_kvidx(self.frame_len, self.world_size)

        self.vae.model.eval()
        # 'wan' has no registry entry to build: it decodes through the very VAE
        # object constructed above, which also does encode. So its streaming
        # wrapper goes around THAT instance. Decide before compiling -- the two
        # paths have different entry points.
        # Not under sequence parallelism: `decode` splits 1D there while
        # `cached_decode_withflag` only has a 2D-grid path, so the streaming
        # version would not be the same computation.
        stream_wan = (
            args.vae == "wan"
            and soulx_runtime.stream_decode_enabled()
            and not self.use_dist
        )
        # Only the full Wan VAE's decode is worth compiling; the tiny decoders
        # are ~32ms and replace this call site outright.
        if args.compile != "off" and args.vae == "wan":
            if stream_wan:
                self.vae.cached_decode_withflag = torch.compile(
                    self.vae.cached_decode_withflag
                )
            else:
                self.vae.decode = torch.compile(self.vae.decode)

        # Swap the decoder AFTER any torch.compile of the real one. self.vae.encode
        # is untouched either way -- the reference image still goes through the
        # full Wan VAE, only the per-chunk decode is replaced.
        decode_fn = soulx_runtime.build_decode_fn(args.vae, self.device)
        if decode_fn is None and stream_wan:
            decode_fn = soulx_runtime.StreamingDecode(self.vae)
        self.fast_dec = None
        if decode_fn is not None:
            self.fast_dec = decode_fn
            self.vae.decode = decode_fn
        # A streaming decoder keeps the VAE's causal cache across chunks, so it
        # takes ONLY the new latents and returns a full T*4 frames. Windowing it
        # would decode the overlap twice and emit 9 frames too many.
        self.stream_dec = getattr(decode_fn, "streaming", False)
        if self.rank == 0:
            spec = soulx_runtime.DECODERS[args.vae]
            how = "streaming causal cache" if self.stream_dec else "3-latent window"
            print(f"[decoder] {spec.label} ({how}) -- {spec.note}", flush=True)

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
            for layer in self.kv_cache[i]:
                self.kv_cache[i][layer]["k"].zero_()
                self.kv_cache[i][layer]["v"].zero_()

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
            if self.stream_dec:
                self.fast_dec.reset()
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
                    if self.stream_dec or it == 0:
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

    def _encode_reference(self, img_path):
        """CLIP + VAE conditioning for a reference image.

        Both outputs are handed to the model fresh on EVERY chunk (arg["y"],
        arg["clip_fea"]) and are never folded into the KV cache, which is what
        lets /anchor swap the reference mid-session without restarting.

        `WanVAE_.encode` opens with `clear_cache()`, which resets the DECODER
        feature map as well as the encoder's. At session start that is harmless
        (the decoder is cold anyway), but mid-session it would silently wipe the
        causal cache StreamingDecode is carrying between chunks -- and since
        StreamingDecode tracks warmth with its own `_first` flag, it would go on
        decoding as though the cache were still warm. So save the decoder half
        and put it back.
        """
        vae_model = getattr(self.vae, "model", None)
        saved = None
        if vae_model is not None and hasattr(vae_model, "_feat_map"):
            saved = (vae_model._feat_map, vae_model._conv_idx)

        image = Image.open(img_path).convert("RGB")
        cond = (
            self.transform(image)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(self.device, torch.bfloat16)
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
            clip_ctx = self.clip.visual(cond)
            y = (
                self.vae.encode(torch.concat([cond, pad], dim=2))
                .to(self.device)
                .unsqueeze(0)
            )
        y = torch.concat([msk, y], dim=1)

        if saved is not None:
            vae_model._feat_map, vae_model._conv_idx = saved
        return clip_ctx, y[:, :, : self.frame_num_init // 4 + 1, ...]

    # --- the live session ----------------------------------------------------
    def run_session(self, params):
        """Runs on ALL ranks. rank0 additionally drives io/HLS."""
        img_path = params["img_path"]
        main_prompt = params.get("main_prompt") or WORLD_PROMPT

        self._reset_kv()
        clip_ctx, y_cut = self._encode_reference(img_path)
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
                    # keep the newest emitted frame so /anchor can promote it to
                    # the reference image. Copied, not viewed: arr belongs to the
                    # generator and is reused.
                    STATE.last_frame = arr[-1].copy()
                    # WebRTC first: it takes the raw array, so it pays none of
                    # the JPEG encode below, and start_frame carries the media
                    # clock its RTP timestamps are derived from.
                    if WEBRTC is not None:
                        try:
                            WEBRTC.publish_video(arr, start_frame)
                        except Exception as e:
                            print(f"[webrtc video] {e}", flush=True)
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
                    if WEBRTC is not None:
                        try:
                            WEBRTC.publish_audio(pcm_bytes, start_sample)
                        except Exception as e:
                            print(f"[webrtc audio] {e}", flush=True)
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
        # The warmup decodes into the same causal cache; a session must not
        # inherit its state, and a restarted session must not inherit the dead
        # one's -- both are the same reset.
        if self.stream_dec:
            self.fast_dec.reset()
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
                    new_strength = None
                    with STATE.lock:
                        # leave a directive alone when this session is on its way
                        # out: we would consume it here and then break before
                        # applying it, swallowing a pose that /anchor queued for
                        # the session about to replace us
                        if not stop and STATE.pending_directive is not None:
                            directive = STATE.pending_directive
                            STATE.pending_directive = None
                        if STATE.pending_lora_strength is not None:
                            new_strength = STATE.pending_lora_strength
                            STATE.pending_lora_strength = None
                    if new_strength is not None and self.lora is not None:
                        t0 = time.time()
                        self.lora.set_strength(new_strength)
                        STATE.log(
                            "lora",
                            f"strength -> {new_strength} ({time.time() - t0:.1f}s)",
                        )
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
                    payload = {
                        "stop": stop,
                        "directive": directive,
                        "win": win,
                    }
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
                        # per-action override, else the live default
                        hold = d.get("hold")
                        transition_ttl = int(hold) if hold else STATE.action_hold
                    else:
                        transition_prompt = None
                        transition_ttl = 0

                # re-encode only when the composed world+state+transition changes
                tgt = compose_prompt()
                if tgt != cur_prompt_str:
                    cur_ctx = self._encode_text(tgt)
                    cur_prompt_str = tgt

                # publish for /status. Done here rather than at the directive
                # site because this runs every chunk, so it also picks up a
                # transition that expired at the end of the previous one.
                if self.rank == 0:
                    STATE.state_prompt = state_prompt
                    STATE.transition_prompt = transition_prompt
                    STATE.transition_ttl = max(transition_ttl, 0)

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
                    elif self.stream_dec or iteration == 0:
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
                            "ctx": [
                                c.detach().to(torch.bfloat16).cpu() for c in cur_ctx
                            ],
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
                        (
                            _sr_pool.submit(
                                _sr_request, self.sr_url, sr_job[0], sr_job[1], 0
                            ),
                            heard,
                        )
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
                _close_media_queue(vq)
                _close_media_queue(aq)
                if hls_proc is not None:
                    try:
                        hls_proc.wait(timeout=10)
                    except Exception:
                        hls_proc.kill()
                STATE.active = False
                STATE.stop_flag = False
                STATE.state_prompt = ""
                STATE.transition_prompt = None
                STATE.transition_ttl = 0
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
    # ?transport=ws forces the WebSocket path even when WebRTC is up, so a
    # deployment always has a way back to the transport that is known to work on
    # a given network -- WebRTC needs a relay it can actually reach, and that is
    # not something the server can find out for the client.
    want = (request.args.get("transport") or "").lower()
    use_rtc = WEBRTC is not None and want != "ws"
    return render_template(
        "chat.html",
        stream_resolution=f"{engine.width}x{engine.height}",
        stream_w=engine.width,
        stream_h=engine.height,
        webrtc=use_rtc,
        ice_servers=json.dumps(soulx_webrtc.ice_servers()) if use_rtc else "[]",
    )


@app.route("/offer", methods=["POST"])
def webrtc_offer():
    """WebRTC signalling. One shot: SDP offer in, answer out, no trickle ICE.

    The browser gathers candidates before posting, which costs a moment on
    connect and saves needing a signalling channel of our own.
    """
    if WEBRTC is None:
        return jsonify({"error": "webrtc not enabled (start with --webrtc)"}), 400
    body = request.get_json(force=True, silent=True) or {}
    if "sdp" not in body or "type" not in body:
        return jsonify({"error": "expected {sdp, type}"}), 400
    try:
        return jsonify(WEBRTC.offer(body["sdp"], body["type"]))
    except Exception as e:  # noqa: BLE001 -- surface to the page, don't 500
        print(f"[webrtc offer] {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/webrtc/stats")
def webrtc_stats():
    if WEBRTC is None:
        return jsonify({"enabled": False})
    return jsonify({"enabled": True, **WEBRTC.stats()})


@app.route("/character/default")
def default_character():
    with STATE.lock:
        img_path = STATE.current_image_path or engine.args.image
    return send_file(img_path)


def _queue_session_start(img_path, main_prompt):
    """Queue one ordered causal-session swap and return whether it is a restart."""

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
        STATE.current_image_path = img_path
        STATE.current_main_prompt = main_prompt
        STATE.image_revision = time.time_ns()
        STATE.start_params.put({"img_path": img_path, "main_prompt": main_prompt})
        STATE.stop_flag = restarting
    return restarting


@app.route("/session/start", methods=["POST"])
def session_start():
    with STATE.lock:
        img_path = STATE.current_image_path or engine.args.image
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
    restarting = _queue_session_start(img_path, main_prompt)
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
        # optional per-action hold, in chunks (1 chunk = 32 frames = 2s @16fps).
        # Omitted -> STATE.action_hold. A gesture that should linger (a slow
        # stretch) and one that should not (a quick wave) want different values,
        # so the caller can say, rather than everything sharing one default.
        try:
            hold = int(body.get("hold") or 0)
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "hold must be an int"}), 400
        if hold > 0:
            directive["hold"] = min(hold, 120)  # ~4min ceiling; it is a gesture
    with STATE.lock:
        STATE.pending_directive = directive
    STATE.log("action", text or "(idle)")
    return jsonify({"status": "ok", "hold": directive.get("hold", STATE.action_hold)})


@app.route("/chat", methods=["POST"])
def chat():
    msg = (request.get_json(force=True).get("text") or "").strip()
    if not msg:
        return jsonify({"status": "error", "message": "empty"}), 400
    STATE.log("user", msg)
    change_instruction = parse_change_command(msg)
    if change_instruction is not None:
        if not change_instruction:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "usage: /change <describe the new appearance>",
                    }
                ),
                400,
            )
        with STATE.lock:
            if STATE.appearance_busy:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "another appearance change is still running",
                        }
                    ),
                    409,
                )
            STATE.appearance_busy = True
            source_path = STATE.current_image_path or engine.args.image
            main_prompt = STATE.current_main_prompt
        STATE.log("system", "Creating the new appearance…")
        try:
            img_path = appearance_editor.edit(source_path, change_instruction)
            restarting = _queue_session_start(img_path, main_prompt)
        except AppearanceRateLimit as exc:
            STATE.log("error", str(exc))
            return jsonify({"status": "error", "message": str(exc)}), 429
        except AppearanceError as exc:
            STATE.log("error", str(exc))
            return jsonify({"status": "error", "message": str(exc)}), 502
        finally:
            with STATE.lock:
                STATE.appearance_busy = False
        message = (
            "New appearance ready; restarting the live character…"
            if restarting
            else "New appearance ready; starting the live character…"
        )
        STATE.log("system", message)
        return jsonify(
            {
                "status": "ok",
                "say": "",
                "action": None,
                "pose": None,
                "appearance": True,
                "restarting": restarting,
            }
        )
    try:
        d = brain.reply(msg)
    except Exception as e:
        STATE.log("error", f"LLM failed: {e}")
        return jsonify({"status": "error", "message": f"LLM failed: {e}"}), 500
    # action = transient motion; pose = sustained state to keep afterwards (None -> gesture)
    if d.action or d.pose:
        directive = {"transition": d.action, "rest": d.pose}
        # A duration the script asked for ("…, 2s") overrides the live default,
        # so "turn around for a sec" is a beat and not sixteen seconds of back.
        if d.hold:
            directive["hold"] = min(int(d.hold), directions_mod.MAX_HOLD_CHUNKS)
        with STATE.lock:
            STATE.pending_directive = directive
        held = directive.get("hold", STATE.action_hold)
        STATE.log(
            "action",
            (d.action or f"(pose: {d.pose})")
            + (f" · {int(held * directions_mod.CHUNK_SECONDS)}s" if d.action else ""),
        )
    if d.say:
        try:
            wav = tts.synth(d.say, voice=STATE.voice)
            if len(wav):
                STATE.speech_q.put(wav)
        except Exception as e:
            STATE.log("error", f"TTS failed: {e}")
        STATE.log("chano", d.say)
    return jsonify(
        {
            "status": "ok",
            "say": d.say,
            "action": d.action,
            "pose": d.pose,
            "hold": d.hold,
        }
    )


@app.route("/action_hold", methods=["POST"])
def action_hold():
    """Set how long a transient action survives, in chunks (2s each).

    Takes effect on the NEXT action; one already running keeps the TTL it was
    given. Runtime-settable on purpose: how long a gesture should linger is a
    feel judgement that wants a few tries, and a restart costs a ~3min warmup.
    """
    try:
        chunks = int(request.get_json(force=True).get("chunks"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "chunks must be an int"}), 400
    if not 1 <= chunks <= 120:
        return jsonify({"status": "error", "message": "chunks must be 1..120"}), 400
    with STATE.lock:
        STATE.action_hold = chunks
    STATE.log("system", f"action hold -> {chunks} chunks (~{chunks * 2}s)")
    return jsonify({"status": "ok", "chunks": chunks, "seconds": chunks * 2})


@app.route("/anchor", methods=["POST"])
def anchor():
    """Promote the frame she is in RIGHT NOW to the reference image.

    A pose typed into /action only ever composes onto the T5 prompt, while the
    reference image is re-imposed every chunk -- so where the two disagree the
    reference wins and she slides back to it. That is the "it regressed to the
    idle pic" complaint, and no prompt wording or LoRA strength fixes it,
    because the I2V reference path (img_emb, cross_attn.k_img/v_img) is not
    reachable from either.

    Anchoring inverts it: get her into the pose however you like, then make THAT
    frame the thing being re-imposed. The pose stops being something the prompt
    has to win every chunk and becomes the resting state.

    This restarts the causal session, and it has to. Rebinding the per-chunk
    conditioning (y/clip_fea) instead was tried and measured: seamless, 0.35s,
    no stall or artifacts -- and useless, because she reverted to the old pose
    within seconds. Pose is carried by the rolling KV cache, not by the
    conditioning arguments, so the cache is exactly what has to be re-seeded.

    The restart is cheap: ~3-4s with the model resident (1.8s to finish the
    current chunk, ~0.9s to swap, ~1.2s for the first chunk). It is /change
    that feels slow, and that is its external image-edit API call, not this.

    The sustained state prompt is KEPT, not cleared. Measured the hard way: the
    reference image governs identity, wardrobe and scene, but NOT posture -- she
    relaxes to a neutral resting pose within ~15s whatever the reference shows,
    even when it is a frame of her with both arms raised. What actually holds a
    pose is the state text, which held one for 24s+ in an earlier recording. So
    clearing it here (the obvious-looking move, since the picture "already shows"
    the pose) is precisely what un-pins the pose.

    Body: {"pose": str|null}. Omitted -> whatever state is currently held stays
    held. Pass a pose to overwrite it, or "" to deliberately clear it.
    """
    frame = STATE.last_frame
    if frame is None:
        return jsonify(
            {"status": "error", "message": "no frame yet; start the session first"}
        ), 409
    body = request.get_json(silent=True) or {}
    pose = body.get("pose")

    img_path = os.path.join(BASE_DIR, "uploads", f"anchor_{time.time_ns()}.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    Image.fromarray(frame).save(img_path)
    _prune_uploads(os.path.dirname(img_path), keep=5)

    with STATE.lock:
        main_prompt = STATE.current_main_prompt
        # carry the held state across the restart: the loop starts a new session
        # with state_prompt="" and picks the directive up on its first chunk, so
        # without this the pose would be dropped by the very act of pinning it
        rest = STATE.state_prompt if pose is None else pose.strip()
        STATE.pending_directive = {"transition": None, "rest": rest}
    _queue_session_start(img_path, main_prompt)
    STATE.log("system", "pinned her current pose as the new resting pose")
    return jsonify({"status": "ok", "pose": pose or ""})


@app.route("/lora", methods=["POST"])
def lora_strength():
    """Change the merged LoRA strength without a restart.

    Applied at the next chunk boundary by the generation loop, because the
    merge rewrites the bf16 master weights and forces an fp8 requantize --
    doing that under a running forward would tear the weights mid-chunk.
    A full warmup is ~10 minutes, so this is what makes a strength sweep
    cost one launch instead of one per point.
    """
    if engine.lora is None:
        return jsonify({"status": "error", "message": "no LoRA loaded"}), 400
    if engine.use_dist:
        # Each rank holds its own copy of the weights; a rank-0-only merge
        # would silently desync them. Would need a broadcast at the barrier.
        return jsonify(
            {"status": "error", "message": "strength swap is single-GPU only"}
        ), 400
    try:
        value = float(request.get_json(force=True).get("strength"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "strength must be a number"}), 400
    with STATE.lock:
        STATE.pending_lora_strength = value
    if not STATE.active:
        # nothing is generating, so nobody will reach the chunk boundary
        engine.lora.set_strength(value)
        with STATE.lock:
            STATE.pending_lora_strength = None
        return jsonify({"status": "ok", "strength": value, "applied": "now"})
    return jsonify({"status": "ok", "strength": value, "applied": "next chunk"})


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
            "action_hold": STATE.action_hold,
            "pose": {
                "state": STATE.state_prompt,
                "transition": STATE.transition_prompt,
                "transition_left": STATE.transition_ttl,
            },
            "appearance": {
                "busy": STATE.appearance_busy,
                "configured": appearance_editor.configured,
                "image_url": f"/character/default?v={STATE.image_revision}",
            },
            "lora": (
                None
                if engine.lora is None
                else {
                    "name": engine.lora.name,
                    "strength": engine.lora.strength,
                    "modules": engine.lora.n_modules,
                }
            ),
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
    _paths = soulx_runtime.paths()
    parser = argparse.ArgumentParser(
        description="SoulX-LiveAct interactive persona demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="decoders available to --vae:\n" + soulx_runtime.decoder_help(),
    )
    # Paths default off SOULX_ROOT (see soulx_runtime.Paths) so the same command
    # line works on any box; pass them explicitly only to override the layout.
    parser.add_argument("--ckpt_dir", type=str, default=_paths.ckpt_dir)
    parser.add_argument("--wav2vec_dir", type=str, default=_paths.wav2vec_dir)
    parser.add_argument(
        "--image",
        type=str,
        default=_paths.default_image,
        help="default reference image",
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
    parser.add_argument(
        "--size",
        type=str,
        default="368x640",
        help=f"WIDTHxHEIGHT or a preset name ({', '.join(soulx_runtime.RESOLUTIONS)}); "
        f"both dimensions multiples of 16",
    )
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--fp8_kv_cache", action="store_true", default=False)
    parser.add_argument(
        "--vae",
        type=str,
        default="taew2_1",
        choices=list(soulx_runtime.DECODERS),
        help="decoder for the per-chunk latents: 'wan' is the full Wan2.1 VAE "
        "(452ms, the quality reference), the rest are tiny decoders. "
        "vae.encode always stays the Wan VAE.",
    )
    parser.add_argument(
        "--webrtc",
        action="store_true",
        help="also serve the stream over WebRTC (H.264/Opus), with RTP "
        "timestamps taken from the generator's media clock. Measured against "
        "the MJPEG-over-WebSocket path at 368x640: 0.97 vs 4.47 Mbps, and "
        "inter-frame p99 137ms vs 1603ms. Both transports run off the same "
        "emit worker, so a session can serve them side by side.",
    )
    parser.add_argument(
        "--sr_url",
        type=str,
        default=None,
        help="UltraFlash SR sidecar base URL; omit to stream at native res",
    )
    parser.add_argument("--sr_scale", type=int, default=2)
    parser.add_argument(
        "--action_hold",
        type=int,
        default=8,
        help="chunks a transient action is held before the sustained state "
        "takes back over. One chunk is 32 frames = 2s at 16fps, so the default "
        "8 is ~16s. Change it live with POST /action_hold, or per-gesture with "
        "the 'hold' field on POST /action.",
    )
    parser.add_argument(
        "--fp8",
        choices=["off", "blocks", "all"],
        default="blocks",
        help="which linears the vLLM W8A8 GEMM covers: 'blocks' = attention/FFN "
        "matmuls only (leaves modulation/embeddings/head in bf16), 'all' = every "
        "linear and KNOWN TO PRODUCE NOISE, 'off' = bf16. Auto-disabled below sm_89.",
    )
    parser.add_argument(
        "--attn",
        choices=["auto", "sdpa", "fa3", "sage"],
        default="auto",
        help="self-attention backend; 'auto' picks FA3 on Hopper and SDPA "
        "elsewhere. 'sage' is SageAttention (int8 QK / fp8 PV) and is never "
        "picked by auto: on sm90 its Hopper kernel is numerically broken and "
        "the accurate ones are slower than SDPA. The kernel is verified against "
        "SDPA at first use and falls back rather than streaming noise.",
    )
    parser.add_argument(
        "--sage_scope",
        default="self",
        help="which attentions sage covers: 'self' (default), 'all', or a "
        "combination like 'self+cross'. Same reasoning as --fp8: cross-attention "
        "(257 image + 512 text tokens) and audio cross-attention (~32 tokens) "
        "carry almost no FLOPs but drive identity and lipsync.",
    )
    parser.add_argument(
        "--compile",
        choices=["off", "blocks", "full"],
        default="blocks",
        help="torch.compile strategy: 'blocks' compiles each transformer block "
        "independently (Pruna Wan recipe, 17-20%%, survives graph breaks), "
        "'full' compiles the whole streaming model, 'off' disables it. "
        "Warmup is minutes and is re-paid on every resolution change.",
    )
    parser.add_argument(
        "--compile_mode",
        choices=["default", "reduce-overhead", "max-autotune-no-cudagraphs"],
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument(
        "--lora",
        default=os.environ.get("SOULX_LORA", ""),
        help="path to a kohya-format LoRA (.safetensors, lora_unet_* keys) to "
        "merge into the DiT at load time. Merged, not run as an adapter, so "
        "s/chunk and VRAM stay comparable to a no-LoRA run. The strength can "
        "be changed at runtime via POST /lora without a restart.",
    )
    parser.add_argument(
        "--lora_strength",
        type=float,
        default=float(os.environ.get("SOULX_LORA_STRENGTH", "1.0")),
        help="LoRA merge strength (default 1.0). The DiT is distilled to a "
        "4-step schedule, so a LoRA trained for many-step inference usually "
        "wants a value well below 1.0; sweep it via POST /lora.",
    )
    parser.add_argument(
        "--autostart", action="store_true", help="start session immediately"
    )
    args = parser.parse_args()
    STATE.action_hold = args.action_hold

    # Separate FIFOs/segments for simultaneous debug or A/B processes. Sharing
    # hls_output/live made one port open another port's FIFO and deadlocked a
    # character restart while its media queues filled.
    if "SOULX_HLS_ROOT" not in os.environ:
        HLS_ROOT = os.path.join(BASE_DIR, "hls_output", str(args.port))
    os.makedirs(HLS_ROOT, exist_ok=True)

    if not args.image:
        raise SystemExit(
            f"no reference image: pass --image, or put one in {_paths.assets}"
        )
    missing = _paths.missing()
    if missing and not os.path.exists(args.ckpt_dir):
        raise SystemExit("missing model files:\n  " + "\n  ".join(missing))

    engine = LiveEngine(args)
    if engine.rank == 0:
        if args.webrtc:
            _enable_webrtc(args.fps)
        tts = TTS(args.aux_device, args.aux_url)
        brain = Brain(args.aux_device, args.aux_url)
        STATE.current_image_path = args.image
        STATE.image_revision = time.time_ns()
        threading.Thread(target=control_loop_rank0, daemon=True).start()
        if args.autostart:
            _queue_session_start(args.image, WORLD_PROMPT)
        print(
            f"\n LiveAct interactive demo on http://0.0.0.0:{args.port}\n", flush=True
        )
        app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)
    else:
        control_loop_other()
    if dist.is_initialized():
        dist.destroy_process_group()
