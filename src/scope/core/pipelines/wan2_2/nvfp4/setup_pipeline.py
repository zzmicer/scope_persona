# SPDX-License-Identifier: Apache-2.0
#
# Modified from upstream NVlabs/LongLive (utils/inference_utils.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
#
# Vendored into Daydream Scope for the NVFP4 (W4A4) inference path. The
# ``setup_nvfp4_pipeline`` orchestrator is ported verbatim; only the import
# paths are rewritten to point at this subpackage. The Daydream Scope
# longlive2 pipeline calls ``setup_nvfp4_pipeline`` to quantize the 5B
# generator and load the NVFP4 checkpoints.
"""NVFP4 pipeline setup: detect checkpoint type, quantize, load weights, wire KV-quant."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .checkpoint import (
    clean_fsdp_state_dict_keys,
    drop_fouroversix_master_weights,
    is_nvfp4_state_dict,
    is_te_nvfp4_checkpoint,
    quantize_model_for_fouroversix_nvfp4,
    quantize_model_for_transformer_engine_nvfp4,
    unwrap_generator_state_dict,
)


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_generator_checkpoint(
    generator,
    checkpoint_path: str,
    *,
    use_ema: bool = False,
    strict: bool | None = None,
):
    """Load a LongLive generator checkpoint into ``generator``."""
    checkpoint = _torch_load(checkpoint_path)
    state_dict = unwrap_generator_state_dict(checkpoint, use_ema=use_ema)
    if use_ema:
        state_dict = clean_fsdp_state_dict_keys(state_dict)
    if strict is None:
        strict = not use_ema
    return generator.load_state_dict(state_dict, strict=strict)


def _load_lora_state_dict(lora_ckpt_path: str) -> Mapping[str, torch.Tensor]:
    """Load a LoRA checkpoint, unwrapping ``generator_lora`` when present."""
    checkpoint = _torch_load(lora_ckpt_path)
    if isinstance(checkpoint, Mapping) and "generator_lora" in checkpoint:
        return checkpoint["generator_lora"]
    return checkpoint


def apply_and_merge_lora(
    pipeline,
    config,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
):
    """Wrap ``pipeline.generator.model`` with a LoRA adapter, load weights, and merge.

    The merged module ends up structurally identical to the original generator
    (``nn.Linear`` layers carrying the base + LoRA delta), which is what NVFP4
    quantization needs as its starting point.

    Returns ``True`` when LoRA was applied and merged, ``False`` when the config
    did not request a LoRA adapter.

    ``peft`` and the upstream ``configure_lora_for_model`` helper are imported
    lazily; the latter is not vendored into Scope, so this branch only runs on
    the pod when a ``lora_ckpt`` is supplied (the released NVFP4 yaml does not
    set one).
    """
    adapter_cfg = getattr(config, "adapter", None)
    lora_ckpt = getattr(config, "lora_ckpt", None)
    if adapter_cfg is None or not lora_ckpt:
        return False

    import peft
    from utils.lora_utils import configure_lora_for_model

    if device is not None:
        pipeline.generator.to(device=torch.device(device), dtype=dtype)
    else:
        pipeline.generator.to(dtype=dtype)

    if verbose:
        print(f"[LoRA] Wrapping generator with adapter config: {adapter_cfg}")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=adapter_cfg,
        is_main_process=verbose,
    )

    if verbose:
        print(f"[LoRA] Loading LoRA weights from: {lora_ckpt}")
    lora_state = _load_lora_state_dict(lora_ckpt)
    peft.set_peft_model_state_dict(pipeline.generator.model, lora_state)  # type: ignore[arg-type]

    if verbose:
        print("[LoRA] Merging LoRA delta into base weights (merge_and_unload)...")
    pipeline.generator.model = pipeline.generator.model.merge_and_unload(
        safe_merge=True
    )
    pipeline.generator.model.eval().requires_grad_(False)
    pipeline.is_lora_enabled = False
    pipeline.is_lora_merged = True
    return True


def place_vae_for_streaming(pipeline, config) -> torch.device | None:
    """Move ``pipeline.vae`` to ``config.vae_device`` for streaming-pipeline decode.

    Only acts when both ``streaming_vae`` and ``vae_device`` are set; otherwise
    leaves the VAE on whatever device the rest of the pipeline already uses.
    Mirrors the relocation done in ``inference.py`` so that quick-start scripts
    can opt in to the streaming-pipeline VAE simply by enabling those config
    fields.
    """
    if not bool(getattr(config, "streaming_vae", False)):
        return None
    vae_device_str = getattr(config, "vae_device", None)
    if not vae_device_str:
        return None

    vae_device = torch.device(vae_device_str)
    pipeline.vae.to(device="cpu")
    pipeline.vae.to(device=vae_device)
    if hasattr(pipeline.vae, "mean"):
        pipeline.vae.mean = pipeline.vae.mean.to(device=vae_device)
        pipeline.vae.std = pipeline.vae.std.to(device=vae_device)
    return vae_device


def setup_nvfp4_pipeline(
    pipeline,
    config,
    device: torch.device | str,
    *,
    verbose: bool = False,
):
    """Configure ``pipeline`` for NVFP4 inference from a generator checkpoint.

    Handles both supported NVFP4 backends:

    * ``model_quant_use_transformer_engine=True`` -> a BF16 generator checkpoint
      (``model_te.pt``) that gets wrapped with TransformerEngine NVFP4 modules and
      materialized after moving to ``device``.
    * ``model_quant_use_transformer_engine=False`` -> either a BF16 generator
      checkpoint that gets quantized with FourOverSix at load time, or a
      pre-materialized FourOverSix NVFP4 state dict (``model_4o6.pt``) loaded
      directly into the already-quantized architecture.

    Optional LoRA support (BF16 base only): when ``config.adapter`` and
    ``config.lora_ckpt`` are both set, the LoRA adapter is loaded on the BF16
    base generator, merged via ``merge_and_unload``, and the resulting weights
    are then quantized — so the same yaml can swap between TE and FourOverSix
    backends without pre-merging the LoRA checkpoint.

    For materialized FourOverSix checkpoints LoRA cannot be applied (the master
    weights have already been quantized away); ``lora_ckpt``/``adapter`` are
    ignored in that case with a printed warning.

    Returns the (mutated) ``pipeline`` with its quantized generator on ``device``.
    """
    if not bool(getattr(config, "model_quant", False)):
        raise ValueError(
            "setup_nvfp4_pipeline requires model_quant=true in the config."
        )

    generator_ckpt = getattr(config, "generator_ckpt", None)
    if not generator_ckpt:
        raise ValueError("checkpoints.generator_ckpt is required for NVFP4 inference.")

    use_te = bool(getattr(config, "model_quant_use_transformer_engine", False))
    device = torch.device(device)
    use_ema = bool(getattr(config, "use_ema", False))

    checkpoint = _torch_load(generator_ckpt)
    state_dict = unwrap_generator_state_dict(checkpoint, use_ema=use_ema)
    if use_ema:
        state_dict = clean_fsdp_state_dict_keys(state_dict)

    if is_te_nvfp4_checkpoint(checkpoint):
        raise ValueError(
            "Detected a TransformerEngine module state_dict export (no longer supported). "
            "Re-export with `--backend transformer_engine` (merged BF16) or `--backend fouroversix`."
        )

    is_prequantized = is_nvfp4_state_dict(state_dict)
    has_lora_request = bool(getattr(config, "adapter", None)) and bool(
        getattr(config, "lora_ckpt", None)
    )

    pipeline.is_lora_enabled = False
    pipeline.is_lora_merged = False

    if is_prequantized:
        if has_lora_request and verbose:
            print(
                "[NVFP4] generator_ckpt is a materialized FourOverSix NVFP4 checkpoint; "
                "ignoring lora_ckpt/adapter because the master weights are already quantized. "
                "Use a BF16 base checkpoint if you need to load a LoRA on top."
            )
        if use_te:
            raise ValueError(
                "generator_ckpt is a materialized NVFP4 (FourOverSix) checkpoint; set "
                "model_quant_use_transformer_engine: false."
            )
        pipeline.generator.model, _ = quantize_model_for_fouroversix_nvfp4(
            pipeline.generator.model,
            config=config,
            keep_master_weights=False,
            verbose=verbose,
        )
        drop_fouroversix_master_weights(pipeline.generator.model)
        pipeline.generator.load_state_dict(state_dict, strict=True)

        pipeline.text_encoder.to(dtype=torch.bfloat16)
        pipeline.vae.to(dtype=torch.bfloat16)
    else:
        load_strict = not use_ema
        pipeline.generator.load_state_dict(state_dict, strict=load_strict)

        if has_lora_request:
            # Apply + merge LoRA on the BF16 base before quantization. Move the
            # generator to CUDA first so the TE wrapper (which requires CUDA
            # modules) can later replace the merged Linear layers in-place.
            apply_and_merge_lora(
                pipeline,
                config,
                device=device,
                dtype=torch.bfloat16,
                verbose=verbose,
            )

        if use_te:
            pipeline.generator.model, _ = quantize_model_for_transformer_engine_nvfp4(
                pipeline.generator.model,
                config=config,
                keep_master_weights=False,
                verbose=verbose,
            )
            te_fallback = bool(
                getattr(config, "model_quant_te_fallback_to_fouroversix", False)
            )
            if te_fallback:
                from .quant import (
                    _materialize_mixed_quantized_weights_for_inference as materialize_fn,
                )
            else:
                from .quant import (
                    _materialize_transformer_engine_weights_for_inference as materialize_fn,
                )
        else:
            pipeline.generator.model, _ = quantize_model_for_fouroversix_nvfp4(
                pipeline.generator.model,
                config=config,
                keep_master_weights=False,
                verbose=verbose,
            )
            from .quant import (
                _materialize_quantized_weights_for_inference as materialize_fn,
            )

        # NOTE: ``pipeline`` is a lightweight namespace exposing .generator /
        # .text_encoder / .vae (see LongLive2Pipeline.__init__), NOT an nn.Module,
        # so cast each component individually instead of pipeline.to(...).
        pipeline.generator.to(dtype=torch.bfloat16)
        pipeline.text_encoder.to(dtype=torch.bfloat16)
        pipeline.vae.to(dtype=torch.bfloat16)
        materialize_fn(pipeline.generator.model, target_device=device)

    pipeline.generator.to(device=device)
    pipeline.text_encoder.to(device=device)
    pipeline.vae.to(device=device)
    place_vae_for_streaming(pipeline, config)

    return pipeline
