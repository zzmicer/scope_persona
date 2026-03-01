"""Mixin for pipelines that support Lynx IP-Adapter (face identity preservation).

This mixin handles upfront IP-Adapter loading at pipeline initialization,
treating it as a toggle similar to the VACE mixin. Pipelines using this mixin
can enable IP-Adapter by loading weights at initialization time.

Composition Order:
When both VACE and IP-Adapter are present, they must be applied in the correct order:
  1. VACE wraps base model: VACE(CausalWan)
  2. IP-Adapter wraps VACE: IPAdapter(VACE(CausalWan))
  3. LoRA wraps outermost: PeftModel(IPAdapter(VACE(CausalWan)))

This ensures:
- VACE hint injection happens inside blocks (via BaseWanAttentionBlock)
- IP-Adapter params flow through kwargs to the same blocks
- LoRA modifies the outermost weights
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Get value from config whether it's dict-like or object-like."""
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class IPAdapterEnabledPipeline:
    """Shared IP-Adapter integration for Wan2.1-based pipelines.

    Pipelines using this mixin are expected to:
    - Call `_init_ip_adapter(config, model, device, dtype)` during __init__
      to wrap the model with IP-Adapter support if ip_adapter_path is provided.

    The mixin handles:
    - Creating the Perceiver Resampler
    - Wrapping the model with IPAdapterWanModel
    - Loading resampler + IP layer weights from checkpoint
    - Moving components to the correct device/dtype

    The mixin keeps track of:
    - self.ip_adapter_enabled: Whether IP-Adapter is active
    - self.ip_adapter_resampler_path: Path to resampler weights
    - self.ip_adapter_ip_layers_path: Path to IP cross-attention weights
    """

    ip_adapter_enabled: bool = False
    ip_adapter_resampler_path: str | None = None
    ip_adapter_ip_layers_path: str | None = None

    def _init_ip_adapter(
        self,
        config: Any,
        model: Any,
        device,
        dtype,
    ) -> Any:
        """Initialize IP-Adapter if paths are provided in config.

        Args:
            config: Pipeline configuration that may contain:
                - 'ip_adapter_resampler_path': Path to resampler.safetensors
                - 'ip_adapter_ip_layers_path': Path to ip_layers.safetensors
            model: Underlying model (CausalWanModel or VACE-wrapped version)
            device: Target device
            dtype: Target dtype

        Returns:
            Model instance, possibly wrapped with IPAdapterWanModel.
        """
        from .models.ip_adapter_model import IPAdapterWanModel
        from .models.resampler import Resampler
        from .utils.weight_loader import load_ip_layer_weights, load_resampler_weights

        resampler_path = _get_config_value(config, "ip_adapter_resampler_path")
        ip_layers_path = _get_config_value(config, "ip_adapter_ip_layers_path")

        self.ip_adapter_resampler_path = resampler_path
        self.ip_adapter_ip_layers_path = ip_layers_path
        self.ip_adapter_enabled = False

        if not resampler_path or not ip_layers_path:
            logger.info(
                "_init_ip_adapter: No IP-Adapter paths provided, IP-Adapter disabled"
            )
            return model

        logger.debug("_init_ip_adapter: Loading IP-Adapter (Lynx Lite)")

        # Create resampler with Lynx Lite config
        start = time.time()
        resampler = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=16,
            embedding_dim=512,  # ArcFace output dim
            output_dim=2048,  # Wan2.1 text encoder hidden dim
        )

        # Load resampler weights
        load_resampler_weights(resampler, resampler_path)
        resampler = resampler.to(device=device, dtype=dtype)
        resampler.eval()
        logger.info(f"_init_ip_adapter: Loaded resampler in {time.time() - start:.3f}s")

        # Wrap model with IP-Adapter
        start = time.time()
        ip_model = IPAdapterWanModel(
            wrapped_model=model,
            resampler=resampler,
            cross_attention_dim=2048,
        )

        # Load IP cross-attention layer weights
        load_ip_layer_weights(ip_model.ip_cross_attn_layers, ip_layers_path)
        ip_model.ip_cross_attn_layers = ip_model.ip_cross_attn_layers.to(
            device=device, dtype=dtype
        )
        ip_model.ip_cross_attn_layers.eval()

        # Rebuild per-block list after moving to device
        ip_layer_mapping = {
            block_idx: layer_idx
            for layer_idx, block_idx in enumerate(ip_model.ip_layers)
        }
        ip_model._ip_cross_attn_layer_per_block = []
        for i in range(ip_model.num_layers):
            if i in ip_layer_mapping:
                ip_model._ip_cross_attn_layer_per_block.append(
                    ip_model.ip_cross_attn_layers[ip_layer_mapping[i]]
                )
            else:
                ip_model._ip_cross_attn_layer_per_block.append(None)

        logger.info(f"_init_ip_adapter: Loaded IP layers in {time.time() - start:.3f}s")

        self.ip_adapter_enabled = True
        logger.info("_init_ip_adapter: IP-Adapter (Lynx Lite) enabled successfully")

        return ip_model
