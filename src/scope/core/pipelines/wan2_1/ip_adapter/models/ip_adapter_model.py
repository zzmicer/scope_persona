# IP-Adapter model wrapper for Wan2.1 — adds face identity cross-attention.
# Ported from ByteDance Lynx Lite: https://github.com/bytedance/lynx
#
# Wraps any CausalWanModel (or CausalVaceWanModel) to inject IP-Adapter
# cross-attention layers that condition on face identity tokens.
# Follows the same composition pattern as CausalVaceWanModel.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WanRMSNorm(nn.Module):
    """RMSNorm without learnable parameters (elementwise_affine=False)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class IPCrossAttentionLayer(nn.Module):
    """Single IP cross-attention layer injected into a transformer block.

    Computes parallel cross-attention with face identity tokens and adds
    the result (scaled by ip_scale) to the standard text cross-attention output.

    Per-layer params: to_k_ip (cross_dim→hidden) + to_v_ip (cross_dim→hidden).
    """

    def __init__(self, hidden_size, cross_attention_dim, bias=False):
        super().__init__()
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=bias)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=bias)
        self.norm_k = WanRMSNorm(hidden_size)

        # Zero-initialize so IP-Adapter has no effect at start
        nn.init.zeros_(self.to_k_ip.weight)
        nn.init.zeros_(self.to_v_ip.weight)

    def forward(self, q, ip_tokens, num_heads):
        """
        Args:
            q: Query [B, seq_len, hidden_size] (from block's cross-attn q projection)
            ip_tokens: Identity tokens from resampler [B, num_ip_tokens, cross_attention_dim]
            num_heads: Number of attention heads

        Returns:
            IP attention output [B, seq_len, hidden_size]
        """
        b, seq_len, dim = q.shape
        head_dim = dim // num_heads

        ip_k = self.norm_k(self.to_k_ip(ip_tokens))
        ip_v = self.to_v_ip(ip_tokens)

        # Reshape for multi-head attention
        q = q.view(b, seq_len, num_heads, head_dim).transpose(1, 2)
        ip_k = ip_k.view(b, -1, num_heads, head_dim).transpose(1, 2)
        ip_v = ip_v.view(b, -1, num_heads, head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, ip_k, ip_v)
        return out.transpose(1, 2).reshape(b, seq_len, dim)


class IPAdapterWanModel(nn.Module):
    """IP-Adapter wrapper that adds face identity cross-attention to any Wan2.1 model.

    Follows the same composition pattern as CausalVaceWanModel:
    - Wraps an existing model (CausalWanModel or CausalVaceWanModel)
    - Adds IP cross-attention layers to every 2nd transformer block
    - Loads Lynx Lite weights (resampler.safetensors + ip_layers.safetensors)

    IP params flow through the existing kwargs plumbing:
      IPAdapterWanModel._forward_inference()
        → wrapped_model._forward_inference(**block_kwargs)
          → _filter_block_kwargs() auto-indexes per-block ip_cross_attn_layer
            → BaseWanAttentionBlock.forward(ip_tokens=..., ip_cross_attn_layer=...)
    """

    def __init__(
        self,
        wrapped_model,
        resampler,
        ip_layers=None,
        cross_attention_dim=2048,
    ):
        """
        Args:
            wrapped_model: CausalWanModel, CausalVaceWanModel, or any Wan2.1 model
            resampler: Perceiver Resampler module (already loaded with weights)
            ip_layers: List of layer indices to inject IP attention (default: every 2nd)
            cross_attention_dim: Resampler output dim (must match resampler.output_dim)
        """
        super().__init__()

        self.wrapped_model = wrapped_model

        # Extract config via duck typing
        self.num_layers = wrapped_model.num_layers
        self.dim = wrapped_model.dim
        self.num_heads = wrapped_model.num_heads

        # Store resampler as a submodule
        self.resampler = resampler

        # Determine which layers get IP attention
        if ip_layers is None:
            ip_layers = list(range(0, self.num_layers, 2))
        self.ip_layers = ip_layers

        # Create IP cross-attention layers (one per selected block)
        self.ip_cross_attn_layers = nn.ModuleList(
            [
                IPCrossAttentionLayer(
                    hidden_size=self.dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in ip_layers
            ]
        )

        # Build per-block list: ip_cross_attn_layer[i] is the layer for block i,
        # or None if block i doesn't have IP attention.
        # This list has length num_layers so _filter_block_kwargs auto-indexes it.
        ip_layer_mapping = {
            block_idx: layer_idx for layer_idx, block_idx in enumerate(ip_layers)
        }
        self._ip_cross_attn_layer_per_block = []
        for i in range(self.num_layers):
            if i in ip_layer_mapping:
                self._ip_cross_attn_layer_per_block.append(
                    self.ip_cross_attn_layers[ip_layer_mapping[i]]
                )
            else:
                self._ip_cross_attn_layer_per_block.append(None)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        ip_image_embed=None,
        ip_scale=1.0,
        **kwargs,
    ):
        """Forward pass with optional IP-Adapter identity conditioning.

        If ip_image_embed is provided, it gets processed through the resampler
        and injected via cross-attention at selected layers.
        """
        # Encode face embedding through resampler
        ip_tokens = None
        if ip_image_embed is not None:
            ip_tokens = self.resampler(ip_image_embed)  # [B, 16, 2048]

        # Pass IP params as block_kwargs. They flow through:
        # wrapped_model._forward_inference(**block_kwargs)
        #   → _filter_block_kwargs() sees ip_cross_attn_layer as list[num_layers]
        #     → auto-indexes to per-block value
        #   → BaseWanAttentionBlock.forward(ip_tokens=..., ip_cross_attn_layer=...)
        return self.wrapped_model._forward_inference(
            x,
            t,
            context,
            seq_len,
            ip_tokens=ip_tokens,
            ip_scale=ip_scale,
            ip_cross_attn_layer=self._ip_cross_attn_layer_per_block,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self.wrapped_model._forward_train(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped_model, name)
