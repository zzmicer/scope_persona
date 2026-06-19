"""OmniForcing / LTX-2 runtime adapter.

This module isolates every import of the external OmniForcing LTX packages
(``ltx_core``, ``ltx_causal``, ``ltx_distillation``) behind lazy functions so that
``scope.core.pipelines.omniforcing`` is import-safe on hosts that do not have the
LTX stack installed (e.g. macOS / CI) — the registry must be able to import the
pipeline module to read its config, and the contract tests run on CPU.

The LTX packages are NOT vendored into this repo (LTX-2 Community License +
~tens of thousands of lines). They are installed on the GPU host per
``docs/usage.md``. All OmniForcing-specific construction and the autoregressive
audio-video inference loop live here, behind ``build_runtime``.

The construction + inference sequence mirrors the upstream single-file reference
``ltx-distillation/scripts/omniforcing_causal_inference.py`` (verified against the
installed package on an H100 pod, 2026-06-19): build a ``CausalLTXModel`` wrapped
in ``CausalLTX2DiffusionWrapper``, load the LTX-2 base checkpoint (stripping audio
sink tokens) then the distilled OmniForcing generator weights, build the Gemma
text encoder + video/audio VAE wrappers, convert the distilled timestep schedule
to flow-matching sigmas, and drive ``CausalAVInferencePipeline``.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Modules that must be importable for the real (GPU) path to run.
_REQUIRED_MODULES = ("ltx_core", "ltx_causal", "ltx_distillation")

# Audio latent frames per causal block, mirroring ltx_causal.attention.mask_builder
# (AUDIO_FRAMES_FIRST_BLOCK / AUDIO_FRAMES_PER_BLOCK). Block 0 absorbs the causal-fix
# asymmetry (4 video + 26 audio); blocks k>=1 are 3 video + 25 audio.
_AUDIO_FRAMES_FIRST_BLOCK = 26
_AUDIO_FRAMES_PER_BLOCK = 25


def is_available() -> bool:
    """Return True if the OmniForcing LTX runtime is installed on this host.

    Cheap check (does not import torch-heavy modules): only inspects whether the
    package specs can be found. The actual heavy imports happen in
    ``build_runtime``.
    """
    try:
        return all(
            importlib.util.find_spec(name) is not None for name in _REQUIRED_MODULES
        )
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def _resolve_paths(model_dir: str, model_config: Any) -> dict[str, Path]:
    """Resolve on-disk weight paths under ``model_dir``.

    Layout mirrors how scope downloads HF artifacts: ``<model_dir>/<repo-last-
    segment>/<files>``. The OmniForcing generator repo and the Lightricks/LTX-2
    base repo each get their own subdirectory.

    The minimal weight subset (confirmed on the pod) is:
      - ``LTX-2/ltx-2-19b-dev.safetensors`` — consolidated base: transformer +
        video VAE + audio VAE + vocoder + connectors.
      - ``LTX-2/text_encoder/`` + ``LTX-2/tokenizer/`` — the Gemma-3 text encoder.
        The loader (``ModelLedger`` / ``module_ops_from_gemma_root``) does a
        recursive ``rglob`` under ``gemma_root`` for ``tokenizer.model``,
        ``preprocessor_config.json`` and ``model*.safetensors``, so ``gemma`` must
        point at the LTX-2 *root* (which contains both subdirs), not at
        ``text_encoder/``.
      - ``omniforcing-ltx2-5s-causal/*.safetensors`` (+ index) — distilled gen.
    The per-component diffusers dirs (``vae/``, ``audio_vae/``, ``vocoder/``,
    ``connectors/``) are NOT needed — they live inside the consolidated base.
    """
    root = Path(model_dir)
    base_dir = root / "LTX-2"
    gen_dir = root / "omniforcing-ltx2-5s-causal"
    base_ckpt = getattr(model_config, "base_checkpoint", "ltx-2-19b-dev.safetensors")
    return {
        "base_checkpoint": base_dir / base_ckpt,
        "generator_index": gen_dir
        / "omniforcing_ltx2_5s_causal.safetensors.index.json",
        "generator_dir": gen_dir,
        # Gemma-3 root: the LTX-2 base dir (recursive lookup finds text_encoder/
        # + tokenizer/ under it).
        "gemma": base_dir,
    }


def build_runtime(
    *,
    model_dir: str,
    model_config: Any,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> OmniForcingRuntime:
    """Construct the full OmniForcing AV runtime (GPU host only).

    Loads the causal generator (base LTX-2 weights + distilled OmniForcing
    weights), the Gemma text encoder, the video + audio VAEs and the vocoder, and
    wraps them in the autoregressive ``CausalAVInferencePipeline``.

    Raises ``RuntimeError`` if the LTX runtime is not installed.
    """
    if not is_available():
        raise RuntimeError(
            "OmniForcing LTX runtime is not installed. Install ltx-core, "
            "ltx-causal and ltx-distillation on the GPU host (see "
            "src/scope/core/pipelines/omniforcing/docs/usage.md)."
        )
    # Heavy imports are deferred to here so module import stays cheap/safe.
    return OmniForcingRuntime(
        paths=_resolve_paths(model_dir, model_config),
        model_config=model_config,
        config=config,
        device=device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Checkpoint / schedule helpers (mirrored from omniforcing_causal_inference.py).
# Kept here so the runtime is self-contained and does not depend on the upstream
# script (which is a CLI entry point, not an importable module).
# ---------------------------------------------------------------------------


def _load_safetensors_shards(paths: list[Path]) -> dict[str, Any]:
    from safetensors.torch import load_file

    state_dict: dict[str, Any] = {}
    for path in paths:
        state_dict.update(load_file(str(path)))
    return state_dict


def _load_checkpoint_state_dict(path: str, prefer_ema: bool = False) -> dict[str, Any]:
    """Load a (possibly sharded) LTX checkpoint and return its state dict."""
    import json

    import torch

    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        index_files = sorted(checkpoint_path.glob("*.safetensors.index.json"))
        if index_files:
            return _load_checkpoint_state_dict(str(index_files[0]), prefer_ema)
        shard_files = sorted(checkpoint_path.glob("*.safetensors"))
        if shard_files:
            return _load_safetensors_shards(shard_files)
        raise FileNotFoundError(f"No safetensors shards found in {checkpoint_path}")

    if path.endswith(".safetensors.index.json"):
        with open(path, encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        shard_files = [checkpoint_path.parent / name for name in shard_names]
        return _load_safetensors_shards(shard_files)

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if prefer_ema and isinstance(checkpoint, dict) and "generator_ema" in checkpoint:
        return checkpoint["generator_ema"]
    for key in ("generator", "model", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key]
    return checkpoint


def _remap_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Map common LTX checkpoint key layouts to CausalLTX2DiffusionWrapper keys."""
    if not state_dict:
        return state_dict

    non_transformer_prefixes = (
        "vae.",
        "audio_vae.",
        "vocoder.",
        "model.vae.",
        "model.audio_vae.",
        "model.vocoder.",
    )
    remapped_non_transformer_prefixes = (
        "model.audio_embeddings_connector.",
        "model.video_embeddings_connector.",
    )

    if any(key.startswith("model.diffusion_model.") for key in state_dict):
        remapped = {}
        for key, value in state_dict.items():
            if not key.startswith("model.diffusion_model."):
                continue
            new_key = "model." + key[len("model.diffusion_model.") :]
            if any(
                new_key.startswith(prefix)
                for prefix in remapped_non_transformer_prefixes
            ):
                continue
            remapped[new_key] = value
        return remapped

    first_key = next(iter(state_dict))
    if first_key.startswith("model.velocity_model."):
        return {
            "model." + key[len("model.velocity_model.") :]: value
            for key, value in state_dict.items()
            if key.startswith("model.velocity_model.")
        }
    if first_key.startswith("model."):
        return {
            key: value
            for key, value in state_dict.items()
            if not any(key.startswith(prefix) for prefix in non_transformer_prefixes)
        }
    return {
        "model." + key: value
        for key, value in state_dict.items()
        if not any(key.startswith(prefix) for prefix in non_transformer_prefixes)
    }


def _add_noise(original: Any, noise: Any, sigma: Any) -> Any:
    """Flow-matching interpolation: x_t = (1 - sigma) * x_0 + sigma * noise."""
    if sigma.dim() == 1:
        sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
    elif sigma.dim() == 2:
        sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
    sigma = sigma.to(dtype=original.dtype)
    return ((1 - sigma) * original + sigma * noise).to(dtype=original.dtype)


def _denoising_sigmas(
    denoising_step_list: list[int],
    num_inference_steps: int,
    device: torch.device,
) -> Any:
    import torch
    from ltx_core.components.schedulers import LTX2Scheduler

    full_sigmas = LTX2Scheduler().execute(steps=num_inference_steps)
    selected = []
    for timestep in denoising_step_list:
        target_sigma = float(timestep) / 1000.0
        idx = (full_sigmas - target_sigma).abs().argmin().item()
        selected.append(full_sigmas[idx])
    return torch.stack(selected).to(device)


def _compute_latent_shapes(
    num_frames: int,
    height: int,
    width: int,
    fps: float = 24.0,
    batch_size: int = 1,
) -> tuple[list[int], list[int]]:
    if (num_frames - 1) % 8 != 0:
        raise ValueError(f"num_frames must be 1 + 8*k, got {num_frames}")
    latent_frames = 1 + (num_frames - 1) // 8
    latent_h = height // 32
    latent_w = width // 32
    # LTX-2 audio latents: 16 kHz / 160 hop / 4 downsample = 25 latent frames/sec.
    video_duration = float(num_frames) / float(fps)
    audio_frames = round(video_duration * 25.0)
    return (
        [batch_size, latent_frames, 128, latent_h, latent_w],
        [batch_size, audio_frames, 128],
    )


class OmniForcingRuntime:
    """Holds the loaded LTX-2 components + the causal AV inference pipeline.

    Constructed only via :func:`build_runtime` on a GPU host. The body performs
    the OmniForcing model loading and exposes :meth:`generate_chunk`, which runs
    one autoregressive pass and returns decoded audio + video for that chunk.
    """

    def __init__(
        self,
        *,
        paths: dict[str, Path],
        model_config: Any,
        config: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        import torch
        from ltx_causal.transformer.causal_model import (
            CausalLTXModel,
            CausalLTXModelConfig,
        )
        from ltx_causal.wrapper import CausalLTX2DiffusionWrapper
        from ltx_core.loader.registry import StateDictRegistry
        from ltx_distillation.inference.causal_pipeline import CausalAVInferencePipeline
        from ltx_distillation.models.text_encoder_wrapper import (
            create_text_encoder_wrapper,
        )
        from ltx_distillation.models.vae_wrapper import create_vae_wrappers

        self.paths = paths
        self.model_config = model_config
        self.config = config
        self.device = device
        self.dtype = dtype

        self.height = int(getattr(config, "height", 512))
        self.width = int(getattr(config, "width", 768))
        self.fps = int(getattr(config, "fps", 24))
        self.audio_sample_rate = int(getattr(config, "audio_sample_rate", 24000))
        self.num_frame_per_block = int(getattr(config, "num_frame_per_block", 3))
        self.num_frame_per_block_first = int(
            getattr(config, "num_frame_per_block_first", 4)
        )
        denoising_steps = list(
            getattr(config, "denoising_steps", None)
            or getattr(model_config, "denoising_steps", None)
            or [1000, 909, 725, 421, 0]
        )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # ---- generator (causal transformer) --------------------------------
        causal_config = CausalLTXModelConfig(
            num_frame_per_block=self.num_frame_per_block,
            num_frame_per_block_first=self.num_frame_per_block_first,
            enable_causal_log_rescale=False,
        )
        model = CausalLTXModel(causal_config).to(device=device, dtype=dtype)
        generator = CausalLTX2DiffusionWrapper(
            model=model,
            video_height=self.height,
            video_width=self.width,
            num_frame_per_block=self.num_frame_per_block,
            num_frame_per_block_first=self.num_frame_per_block_first,
            disable_causal_mask=False,
        )

        base_ckpt = str(paths["base_checkpoint"])
        logger.info("OmniForcing: loading LTX-2 base checkpoint %s", base_ckpt)
        base_sd = _remap_state_dict_keys(_load_checkpoint_state_dict(base_ckpt))
        for key in [k for k in list(base_sd) if "audio_sink_tokens" in k]:
            base_sd.pop(key)
        missing, unexpected = generator.load_state_dict(base_sd, strict=False)
        real_missing = [
            k for k in missing if "mask_builder" not in k and "causal_gate" not in k
        ]
        logger.info(
            "OmniForcing: base load missing=%d unexpected=%d",
            len(real_missing),
            len(unexpected),
        )
        del base_sd

        gen_ckpt = str(paths["generator_index"])
        logger.info("OmniForcing: loading distilled generator %s", gen_ckpt)
        gen_sd = _remap_state_dict_keys(_load_checkpoint_state_dict(gen_ckpt))
        missing, unexpected = generator.load_state_dict(gen_sd, strict=False)
        real_missing = [k for k in missing if "mask_builder" not in k]
        logger.info(
            "OmniForcing: generator load missing=%d unexpected=%d",
            len(real_missing),
            len(unexpected),
        )
        del gen_sd

        generator.requires_grad_(False)
        self.generator = generator.eval()

        # ---- text encoder + VAEs (registry-backed loaders) -----------------
        registry = StateDictRegistry()
        self.text_encoder = create_text_encoder_wrapper(
            checkpoint_path=base_ckpt,
            gemma_path=str(paths["gemma"]),
            device=device,
            dtype=dtype,
            registry=registry,
        ).eval()
        self.video_vae, self.audio_vae = create_vae_wrappers(
            checkpoint_path=base_ckpt,
            device=device,
            dtype=dtype,
            registry=registry,
        )

        # ---- inference pipeline --------------------------------------------
        sigmas = _denoising_sigmas(
            denoising_step_list=denoising_steps,
            num_inference_steps=40,
            device=device,
        )
        logger.info("OmniForcing: denoising_sigmas=%s", sigmas.detach().cpu().tolist())
        self.denoising_sigmas = sigmas
        self.add_noise_fn = _add_noise
        self.context_noise = 0
        self.num_train_timestep = 1000
        # Kept for the non-streaming (one-shot) fallback path + offline use.
        self.pipeline = CausalAVInferencePipeline(
            generator=self.generator,
            add_noise_fn=_add_noise,
            denoising_sigmas=sigmas,
            num_frame_per_block=self.num_frame_per_block,
            num_frame_per_block_first=self.num_frame_per_block_first,
            context_noise=0,
            num_train_timestep=1000,
            clear_cuda_cache_per_round=True,
        )

        # ---- streaming state (persists across generate_chunk calls) --------
        self.streaming = bool(getattr(config, "streaming", True))
        # Windowed-decode tuning (see schema). The video VAE is non-causal, so we
        # decode a sliding latent window with left context + right look-ahead and emit
        # only the interior frames; this both kills the per-block boundary artifacts and
        # bounds the decode cost (constant throughput instead of re-decoding the take).
        self.decode_context_latents = int(getattr(config, "decode_context_latents", 2))
        self.decode_lookahead_latents = int(
            getattr(config, "decode_lookahead_latents", 1)
        )
        self.stream_audio = bool(getattr(config, "stream_audio", False))
        # Snap the requested stream budget (seconds) up to a valid causal block
        # layout: max_video_latent = num_frame_per_block_first + k*num_frame_per_block.
        stream_seconds = float(getattr(config, "stream_max_seconds", 12.0))
        requested_latent = 1 + max(0, round(stream_seconds * self.fps) - 1) // 8
        self._max_video_latent_frames = self._snap_to_block_layout(requested_latent)
        # Audio budget aligned to the video block layout (26 first + 25 per block).
        from ltx_causal.attention.mask_builder import compute_aligned_audio_frames

        self._max_audio_latent_frames = compute_aligned_audio_frames(
            self._max_video_latent_frames,
            num_frame_per_block=self.num_frame_per_block,
            num_frame_per_block_first=self.num_frame_per_block_first,
        )
        logger.info(
            "OmniForcing streaming=%s budget=%.1fs -> max_video_latent=%d "
            "max_audio_latent=%d",
            self.streaming,
            stream_seconds,
            self._max_video_latent_frames,
            self._max_audio_latent_frames,
        )
        self._reset_streaming_state()

    def _snap_to_block_layout(self, n_latent: int) -> int:
        """Round a latent-frame count UP to a valid 4 + k*3 causal block layout."""
        n = max(self.num_frame_per_block_first, int(n_latent))
        remainder = (n - self.num_frame_per_block_first) % self.num_frame_per_block
        if remainder:
            n += self.num_frame_per_block - remainder
        return n

    def _reset_streaming_state(self) -> None:
        """Clear all per-stream state so the next call re-primes from block 0."""
        self._kv_caches = None
        self._cond_dict = None
        self._cached_prompt = None
        self._video_start = 0  # next block's start, in latent frames
        self._audio_start = 0
        self._block_index = 0
        self._video_latent_history = None  # [B, F, 128, h, w] accumulated, on GPU
        self._audio_latent_history = None  # [B, F_a, 128] accumulated, on GPU
        # Number of leading latent frames whose pixels have been emitted (windowed
        # decode bookkeeping). Pixels for a decode of L latents = 1 + 8*(L-1).
        self._decoded_latent_count = 0
        self._emitted_audio_samples = 0
        # Per-block timing accumulators (for periodic throughput logging).
        self._gen_time_acc = 0.0
        self._decode_time_acc = 0.0
        self._timed_blocks = 0

    def generate_chunk(
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        init_cache: bool = False,
    ) -> dict[str, Any]:
        """Produce the next AV chunk and return decoded video + audio.

        Returns the scope AV-dict shape (see pipeline.OmniForcingPipeline):
        ``{video, video_timestamps, audio, audio_sample_rate, audio_timestamps,
        frame_rate}``. ``video`` is ``[T, H, W, C]`` float in ``[0, 1]`` (scope
        convention); ``audio`` is ``[channels, samples]`` float.

        In **streaming** mode (default) each call advances ONE causal block of a
        single continuous take (persistent KV cache), so successive calls yield a
        coherent, non-looping stream rather than independent 5s clips. ``init_cache``
        (re)starts the take from block 0. In non-streaming mode it falls back to a
        full-clip one-shot pass (which loops when called repeatedly).

        Resolution is fixed at construction; the ``height``/``width`` args must match.
        """
        if (height, width) != (self.height, self.width):
            logger.warning(
                "OmniForcing: generate_chunk resolution (%dx%d) differs from the "
                "loaded runtime (%dx%d); using the loaded resolution.",
                height,
                width,
                self.height,
                self.width,
            )

        if self.streaming:
            return self._generate_streaming(
                prompt=prompt, seed=seed, init_cache=init_cache
            )
        return self._generate_oneshot(prompt=prompt, seed=seed, num_frames=num_frames)

    # ------------------------------------------------------------------
    # Streaming (continuous block-by-block) path.
    # ------------------------------------------------------------------

    def _full_sigma(self, sigma: Any, frames: int) -> Any:
        """Broadcast a scalar sigma to a [1, frames] block timestep tensor."""
        return sigma.to(device=self.device, dtype=self.dtype).expand(1, frames)

    def _generate_streaming(
        self, *, prompt: str, seed: int, init_cache: bool
    ) -> dict[str, Any]:
        import torch

        # (Re)start the take: allocate KV caches, re-seed, encode text, reset counters.
        if init_cache or self._kv_caches is None:
            self._start_stream(prompt=prompt, seed=seed)
        # Prompt change mid-stream: re-encode the conditioning but KEEP the cache so
        # the character/scene stays continuous (the new prompt steers from here on).
        elif prompt != self._cached_prompt:
            with torch.no_grad():
                self._cond_dict = self.text_encoder(text_prompts=[prompt])
            self._cached_prompt = prompt

        # Budget guard: the KV cache is a fixed buffer with no sliding window, so once
        # the take fills it we must re-anchor (a visible seam). Documented limitation.
        first = self._block_index == 0
        v_len = self.num_frame_per_block_first if first else self.num_frame_per_block
        if self._video_start + v_len > self._max_video_latent_frames:
            logger.warning(
                "OmniForcing: stream reached the %d-frame KV budget; re-anchoring "
                "(seam). Increase stream_max_seconds for longer takes.",
                self._max_video_latent_frames,
            )
            self._start_stream(prompt=prompt, seed=seed)
            first = True
            v_len = self.num_frame_per_block_first

        import time

        with torch.no_grad():
            t0 = time.time()
            denoised_v, denoised_a = self._generate_block(first=first)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            new_video, new_audio = self._decode_new(denoised_v, denoised_a)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.time()

        # Periodic throughput log (every 10 blocks) to keep diagnosis cheap.
        self._gen_time_acc += t1 - t0
        self._decode_time_acc += t2 - t1
        self._timed_blocks += 1
        if self._timed_blocks >= 10:
            n = self._timed_blocks
            logger.info(
                "OmniForcing stream: %d blocks avg gen=%.3fs decode=%.3fs "
                "(~%.1f fps) latent=%d/%d",
                n,
                self._gen_time_acc / n,
                self._decode_time_acc / n,
                (self.num_frame_per_block * 8.0)
                / max(1e-6, (self._gen_time_acc + self._decode_time_acc) / n),
                self._video_start,
                self._max_video_latent_frames,
            )
            self._gen_time_acc = self._decode_time_acc = 0.0
            self._timed_blocks = 0

        num_new = int(new_video.shape[0])
        # Timestamps are relative to this chunk (scope re-bases per chunk anyway).
        video_timestamps = [i / float(self.fps) for i in range(num_new)]
        audio = None
        audio_timestamps = None
        if new_audio is not None:
            audio = new_audio
            num_samples = int(audio.shape[-1])
            audio_timestamps = [0.0, num_samples / float(self.audio_sample_rate)]

        return {
            "video": new_video,
            "video_timestamps": video_timestamps,
            "audio": audio,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_timestamps": audio_timestamps,
            "frame_rate": self.fps,
        }

    def _start_stream(self, *, prompt: str, seed: int) -> None:
        """Reset state + allocate fresh KV caches and conditioning for a new take."""
        import torch

        self._reset_streaming_state()
        with torch.no_grad():
            self._cond_dict = self.text_encoder(text_prompts=[prompt])
        self._cached_prompt = prompt
        text_seq_len = int(self._cond_dict["video_context"].shape[1])
        gen = self.generator
        while hasattr(gen, "module"):
            gen = gen.module
        self._kv_caches = gen.init_av_kv_caches(
            batch_size=1,
            max_video_frames=self._max_video_latent_frames,
            max_audio_frames=self._max_audio_latent_frames,
            text_seq_len=text_seq_len,
            device=self.device,
            dtype=self.dtype,
        )
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)

    def _generate_block(self, *, first: bool) -> tuple[Any, Any]:
        """Denoise + cache-refresh ONE causal block; returns (denoised_v, denoised_a).

        Mirrors the per-block body of ``CausalAVInferencePipeline.generate`` but for a
        single block, reading/writing the persistent ``self._kv_caches`` at the running
        ``self._video_start``/``self._audio_start`` offsets, then advancing them.
        """
        import torch

        v_len = self.num_frame_per_block_first if first else self.num_frame_per_block
        a_len = _AUDIO_FRAMES_FIRST_BLOCK if first else _AUDIO_FRAMES_PER_BLOCK
        lh, lw = self.height // 32, self.width // 32

        current_video = torch.randn(
            (1, v_len, 128, lh, lw), device=self.device, dtype=self.dtype
        )
        current_audio = torch.randn(
            (1, a_len, 128), device=self.device, dtype=self.dtype
        )

        pred_v = pred_a = None
        for sigma_idx, sigma in enumerate(self.denoising_sigmas[:-1]):
            pred_v, pred_a = self.generator(
                noisy_image_or_video=current_video,
                conditional_dict=self._cond_dict,
                timestep=self._full_sigma(sigma, v_len),
                noisy_audio=current_audio,
                audio_timestep=self._full_sigma(sigma, a_len),
                kv_caches=self._kv_caches,
                current_video_start_frame=self._video_start,
                current_audio_start_frame=self._audio_start,
            )
            next_sigma = self.denoising_sigmas[sigma_idx + 1]
            if float(next_sigma.item()) > 0.0:
                current_video = self.add_noise_fn(
                    pred_v,
                    torch.randn_like(pred_v),
                    self._full_sigma(next_sigma, v_len),
                )
                current_audio = self.add_noise_fn(
                    pred_a,
                    torch.randn_like(pred_a),
                    self._full_sigma(next_sigma, a_len),
                )
            else:
                current_video, current_audio = pred_v, pred_a

        denoised_v, denoised_a = pred_v, pred_a

        # Context-noise refresh: overwrite the cache entries for this block with the
        # clean (context_noise=0) representation the next block expects to read.
        ctx_t = float(self.context_noise) / float(self.num_train_timestep)
        ctx_sigma_v = torch.full(
            (1, v_len), ctx_t, device=self.device, dtype=self.dtype
        )
        ctx_sigma_a = torch.full(
            (1, a_len), ctx_t, device=self.device, dtype=self.dtype
        )
        noisy_ctx_v = (
            self.add_noise_fn(denoised_v, torch.randn_like(denoised_v), ctx_sigma_v)
            if ctx_t > 0.0
            else denoised_v
        )
        noisy_ctx_a = (
            self.add_noise_fn(denoised_a, torch.randn_like(denoised_a), ctx_sigma_a)
            if ctx_t > 0.0
            else denoised_a
        )
        self.generator(
            noisy_image_or_video=noisy_ctx_v,
            conditional_dict=self._cond_dict,
            timestep=ctx_sigma_v,
            noisy_audio=noisy_ctx_a,
            audio_timestep=ctx_sigma_a,
            kv_caches=self._kv_caches,
            current_video_start_frame=self._video_start,
            current_audio_start_frame=self._audio_start,
        )

        self._video_start += v_len
        self._audio_start += a_len
        self._block_index += 1
        return denoised_v, denoised_a

    @staticmethod
    def _px_for_latents(m: int) -> int:
        """Pixel frames produced by decoding the first ``m`` latents of a window.

        The LTX video VAE upsamples time by 8 but the FIRST latent of any decode input
        is a key-frame -> 1 pixel frame; every subsequent latent -> 8. So a decode of
        ``L`` latents yields ``1 + 8*(L-1)`` pixel frames.
        """
        return 0 if m <= 0 else 1 + 8 * (m - 1)

    def _decode_new(self, denoised_v: Any, denoised_a: Any) -> tuple[Any, Any]:
        """Decode the newly-finalized pixel frames for this block (windowed).

        The video VAE is non-causal, so a frame's decode depends on neighbouring latents
        on both sides. We therefore decode a sliding window
        ``[decoded - context, F]`` (left context + the new latents + the look-ahead
        latents), emit only the interior frames up to ``F - lookahead`` (so emitted
        frames have future context), drop the left-context pixels, and hold back the
        look-ahead latents for a later block. This bounds the decode to ~``context +
        block + lookahead`` latents (constant cost) and removes the per-block boundary
        artifacts of decoding growing prefixes.

        Audio is skipped unless ``stream_audio`` is set (the WebRTC transport is
        video-only today, so decoding audio per block is pure waste).
        """
        import torch

        # Accumulate latents (kept for left context; bounded by the stream budget).
        self._video_latent_history = (
            denoised_v
            if self._video_latent_history is None
            else torch.cat([self._video_latent_history, denoised_v], dim=1)
        )
        total = int(self._video_latent_history.shape[1])

        ctx = self.decode_context_latents
        look = self.decode_lookahead_latents
        emit_upto = total - look  # don't emit frames whose future context is unwritten
        empty = torch.empty((0, self.height, self.width, 3), dtype=torch.float32)
        if emit_upto <= self._decoded_latent_count:
            return empty, None  # nothing finalized this block (waiting on look-ahead)

        win_start = max(0, self._decoded_latent_count - ctx)
        window = self._video_latent_history[:, win_start:total]
        video_pixel = self.video_vae.decode_to_pixel(window)  # [1, C, T, H, W]
        video = video_pixel[0]
        if video.shape[0] == 3:  # [C, T, H, W] -> [T, C, H, W]
            video = video.permute(1, 0, 2, 3)
        video = video.permute(0, 2, 3, 1).clamp(0, 1).float().cpu()  # [T, H, W, C]

        # Slice out the interior frames for latents [decoded_count, emit_upto), relative
        # to the window start (which the decoder treats as a key-frame).
        start_px = self._px_for_latents(self._decoded_latent_count - win_start)
        end_px = self._px_for_latents(emit_upto - win_start)
        new_video = video[start_px:end_px]
        self._decoded_latent_count = emit_upto

        new_audio = None
        if self.stream_audio and denoised_a is not None:
            self._audio_latent_history = (
                denoised_a
                if self._audio_latent_history is None
                else torch.cat([self._audio_latent_history, denoised_a], dim=1)
            )
            try:
                waveform = self.audio_vae.decode_to_waveform(self._audio_latent_history)
                audio_full = waveform[0].float().cpu()  # [channels, samples]
                new_audio = audio_full[..., self._emitted_audio_samples :]
                self._emitted_audio_samples = int(audio_full.shape[-1])
            except Exception as exc:  # pragma: no cover - audio decode is best-effort
                logger.warning("OmniForcing: audio decode failed: %s", exc)

        return new_video, new_audio

    # ------------------------------------------------------------------
    # One-shot (legacy / non-streaming) path — regenerates a full clip per call.
    # ------------------------------------------------------------------

    def _generate_oneshot(
        self, *, prompt: str, seed: int, num_frames: int
    ) -> dict[str, Any]:
        import torch

        video_shape, audio_shape = _compute_latent_shapes(
            num_frames=num_frames,
            height=self.height,
            width=self.width,
            fps=self.fps,
            batch_size=1,
        )

        with torch.no_grad():
            conditional_dict = self.text_encoder(text_prompts=[prompt])
            fork_devices = (
                [torch.cuda.current_device()] if self.device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed(seed)
                video_latent, audio_latent = self.pipeline.generate(
                    video_shape=tuple(video_shape),
                    audio_shape=tuple(audio_shape),
                    conditional_dict=conditional_dict,
                )

            video_pixel = self.video_vae.decode_to_pixel(video_latent)
            audio_waveform = None
            if audio_latent is not None:
                try:
                    audio_waveform = self.audio_vae.decode_to_waveform(audio_latent)
                except (
                    Exception
                ) as exc:  # pragma: no cover - audio decode is best-effort
                    logger.warning("OmniForcing: audio decode failed: %s", exc)

        # Normalize video to [T, H, W, C] float in [0, 1].
        video = video_pixel[0]
        if video.shape[0] == 3:  # [C, T, H, W] -> [T, C, H, W]
            video = video.permute(1, 0, 2, 3)
        video = video.permute(0, 2, 3, 1).clamp(0, 1).float().cpu()

        num_video_frames = int(video.shape[0])
        video_timestamps = [i / float(self.fps) for i in range(num_video_frames)]

        audio = None
        audio_timestamps = None
        if audio_waveform is not None:
            audio = audio_waveform[0].float().cpu()  # [channels, samples]
            num_samples = int(audio.shape[-1])
            audio_timestamps = [0.0, num_samples / float(self.audio_sample_rate)]

        del video_latent, audio_latent, conditional_dict
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "video": video,
            "video_timestamps": video_timestamps,
            "audio": audio,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_timestamps": audio_timestamps,
            "frame_rate": self.fps,
        }
