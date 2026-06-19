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
        """Run one autoregressive AV pass and return decoded video + audio.

        Returns the scope AV-dict shape (see pipeline.OmniForcingPipeline):
        ``{video, video_timestamps, audio, audio_sample_rate, audio_timestamps,
        frame_rate}``. ``video`` is ``[T, H, W, C]`` float in ``[0, 1]`` (scope
        convention); ``audio`` is ``[channels, samples]`` float.

        Resolution is fixed at construction (the causal wrapper bakes it in); the
        ``height``/``width`` args must match the loaded config.
        """
        import torch

        if (height, width) != (self.height, self.width):
            logger.warning(
                "OmniForcing: generate_chunk resolution (%dx%d) differs from the "
                "loaded runtime (%dx%d); using the loaded resolution.",
                height,
                width,
                self.height,
                self.width,
            )

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
