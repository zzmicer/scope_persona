"""
Lynx IP-Adapter for Wan2.1 — face identity preservation via cross-attention.

Provides IP-Adapter components for injecting face identity embeddings into
the Wan2.1 diffusion model without constraining pose or motion. Based on
ByteDance's Lynx (Lite variant — ID-Adapter only).

Architecture:
  Reference face image → ArcFace (512d) → Perceiver Resampler (16 tokens @ 2048d)
  → IP cross-attention layers (parallel to text cross-attention in every 2nd block)
  → output = text_attn + ip_scale * ip_attn
"""

from .mixin import IPAdapterEnabledPipeline
from .models.ip_adapter_model import IPAdapterWanModel

__all__ = [
    "IPAdapterEnabledPipeline",
    "IPAdapterWanModel",
]
