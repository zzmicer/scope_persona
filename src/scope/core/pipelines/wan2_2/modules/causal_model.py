# Modified from https://github.com/NVlabs/LongLive (wan_5b/modules/causal_model.py)
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# CausalWanModel for the Wan2.2-TI2V-5B base model (the LongLive 2.0 backbone).
#
# This is a real, complete port. It is structured as Scope's working 1.3B causal
# transformer (../../longlive/modules/causal_model.py) with the architectural
# deltas the upstream 5B reference reveals, verified against:
#   - wan_5b/modules/causal_model.py  (CausalWanSelfAttention, KV-cache rolling,
#                                       sink/local windowing, _forward_inference)
#   - wan_5b/modules/model.py         (rope_params/rope_apply/RMSNorm — identical
#                                       to the 1.3B release linear-RoPE path)
#   - wan_5b/configs/wan_ti2v_5B.py   (dim=3072, ffn_dim=14336, num_heads=24,
#                                       num_layers=30, freq_dim=256, eps=1e-6,
#                                       qk_norm/cross_attn_norm True, vae_stride
#                                       (4,16,16), patch_size (1,2,2))
#
# Deviations from upstream 5B that are deliberate (single-GPU Scope target):
#   * We DROP all sequence-parallel / Ulysses code paths (single GPU).
#   * We DROP the multi-shot RoPE (t_scale / temporal_offset / use_relative_rope)
#     and the NVFP4 quantized-cache / Triton kernels; the release linear path
#     used at inference reduces to the simple RoPE here.
#   * We KEEP Scope's flex_attention block-causal mask + rolling-KV-cache
#     interface (the longlive blocks and wan2_1.utils.initialize_kv_cache drive
#     it) instead of upstream's flash-attn + module-global metadata dict.
#   * Cross-attention uses Scope's cached WanT2VCrossAttention (is_init cache),
#     NOT upstream MultiShotT2VCrossAttention.
#
# Head-dim differs from 1.3B's per-head channel layout only in magnitude
# (head_dim 128 here as well: 3072/24 == 1536/12). The RoPE split formula is
# parametric in head_dim so it is correct for both (see model.py freqs build).
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

# Triton is only required when flex_attention compiles its kernels on GPU.
# Guard so the module imports on CPU/macOS (validation happens later on GPU).
try:
    import triton  # noqa: F401
except ImportError:
    triton = None

from scope.core.pipelines.wan2_1.modules.attention import attention
from .model import (
    WAN_CROSSATTENTION_CLASSES,
    MLPProj,
    WanLayerNorm,
    WanRMSNorm,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)

# flex_attention needs max-autotune on the Wan head/channel configuration to be
# correct (see https://github.com/pytorch/pytorch/issues/133254). The 5B model
# has head_dim=128 with 24 heads; keep the same compile mode as the 1.3B port.
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        # Normalize local_attn_size to an int window count (handles
        # OmegaConf ListConfig / list / tuple coming from config files).
        if not isinstance(local_attn_size, int) and hasattr(
            local_attn_size, "__iter__"
        ):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        self._max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        # `max_attention_size` (the cap on attended tokens) is resolution
        # dependent for the 5B model (the Wan2.2 VAE is 16x spatial vs 8x for
        # the 1.3B VAE, so frame_seqlen differs). Rather than hardcode a
        # frame_seqlen like upstream (which assumes one resolution), we derive
        # it at runtime from the actual grid in forward(). This attribute is a
        # fallback only used if forward() is somehow reached without a grid.
        self.max_attention_size = None
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = s == seq_lens[0].item() * 2
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [
                        roped_query,
                        torch.zeros(
                            [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                            device=q.device,
                            dtype=v.dtype,
                        ),
                    ],
                    dim=1,
                )

                padded_roped_key = torch.cat(
                    [
                        roped_key,
                        torch.zeros(
                            [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                            device=k.device,
                            dtype=v.dtype,
                        ),
                    ],
                    dim=1,
                )

                padded_v = torch.cat(
                    [
                        v,
                        torch.zeros(
                            [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                            device=v.device,
                            dtype=v.dtype,
                        ),
                    ],
                    dim=1,
                )

                attn_out = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask,
                )
                if padded_length > 0:
                    attn_out = attn_out[:, :, :-padded_length]
                x = attn_out.transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [
                        roped_query,
                        torch.zeros(
                            [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                            device=q.device,
                            dtype=v.dtype,
                        ),
                    ],
                    dim=1,
                )

                padded_roped_key = torch.cat(
                    [
                        roped_key,
                        torch.zeros(
                            [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                            device=k.device,
                            dtype=v.dtype,
                        ),
                    ],
                    dim=1,
                )

                padded_v = torch.cat(
                    [
                        v,
                        torch.zeros(
                            [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                            device=v.device,
                            dtype=v.dtype,
                        ),
                    ],
                    dim=1,
                )

                attn_out = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask,
                )
                if padded_length > 0:
                    attn_out = attn_out[:, :, :-padded_length]
                x = attn_out.transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame
            ).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame
            ).type_as(v)

            # Resolution-aware attention-window cap. Upstream hardcodes a
            # frame_seqlen (880 for 5B / 1560 for 1.3B at their reference
            # resolutions); we instead size the window from the runtime
            # frame_seqlen so any supported resolution works. Mirrors
            # upstream's `local_attn_size * frame_seqlen` (or, for global
            # attention, the full local cache budget).
            if self._max_local == -1:
                max_attention_size = kv_cache["k"].shape[1]
            else:
                max_attention_size = self._max_local * frame_seqlen
            self.max_attention_size = max_attention_size

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]

            # Compute cache update parameters without modifying kv_cache directly
            cache_update_info = None
            is_recompute = (
                current_end <= kv_cache["global_end_index"].item() and current_start > 0
            )
            if (
                self.local_attn_size != -1
                and (current_end > kv_cache["global_end_index"].item())
                and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size
                )
            ):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                num_evicted_tokens = (
                    num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                )
                num_rolled_tokens = (
                    kv_cache["local_end_index"].item()
                    - num_evicted_tokens
                    - sink_tokens
                )

                # Compute updated local indices
                local_end_index = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                    - num_evicted_tokens
                )
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation (without modifying the original cache)
                # Create temporary k, v for computation
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()

                # Apply rolling update to the temporary cache
                temp_k[:, sink_tokens : sink_tokens + num_rolled_tokens] = temp_k[
                    :,
                    sink_tokens + num_evicted_tokens : sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
                temp_v[:, sink_tokens : sink_tokens + num_rolled_tokens] = temp_v[
                    :,
                    sink_tokens + num_evicted_tokens : sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()

                # Insert new key/value into the temporary cache
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = (
                    max(local_start_index, sink_tokens)
                    if is_recompute
                    else local_start_index
                )
                if sink_recache_after_switch:
                    write_start_index = local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    temp_k[:, write_start_index:local_end_index] = roped_key[
                        :, roped_offset : roped_offset + write_len
                    ]
                    temp_v[:, write_start_index:local_end_index] = v[
                        :, roped_offset : roped_offset + write_len
                    ]

                # Save cache update info for later use
                cache_update_info = {
                    "action": "roll_and_insert",
                    "sink_tokens": sink_tokens,
                    "num_rolled_tokens": num_rolled_tokens,
                    "num_evicted_tokens": num_evicted_tokens,
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": roped_key[:, roped_offset : roped_offset + write_len],
                    "new_v": v[:, roped_offset : roped_offset + write_len],
                    "current_end": current_end,
                    "is_recompute": is_recompute,
                }

            else:
                # Assign new keys/values directly up to current_end
                local_end_index = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                )
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation (without modifying the original cache)
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = (
                    max(local_start_index, sink_tokens)
                    if is_recompute
                    else local_start_index
                )
                if sink_recache_after_switch:
                    write_start_index = local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    temp_k[:, write_start_index:local_end_index] = roped_key[
                        :, roped_offset : roped_offset + write_len
                    ]
                    temp_v[:, write_start_index:local_end_index] = v[
                        :, roped_offset : roped_offset + write_len
                    ]

                # Save cache update info for later use
                cache_update_info = {
                    "action": "direct_insert",
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": roped_key[:, roped_offset : roped_offset + write_len],
                    "new_v": v[:, roped_offset : roped_offset + write_len],
                    "current_end": current_end,
                    "is_recompute": is_recompute,
                }

            # Use temporary k, v to compute attention
            if sink_tokens > 0:
                # Concatenate sink tokens and local window tokens, keeping total length strictly below max_attention_size
                local_budget = max_attention_size - sink_tokens
                k_sink = temp_k[:, :sink_tokens]
                v_sink = temp_v[:, :sink_tokens]

                if local_budget > 0:
                    local_start_for_window = max(
                        sink_tokens, local_end_index - local_budget
                    )
                    k_local = temp_k[:, local_start_for_window:local_end_index]
                    v_local = temp_v[:, local_start_for_window:local_end_index]
                    k_cat = torch.cat([k_sink, k_local], dim=1)
                    v_cat = torch.cat([v_sink, v_local], dim=1)
                else:
                    k_cat = k_sink
                    v_cat = v_sink
                x = attention(roped_query, k_cat, v_cat)
            else:
                window_start = max(0, local_end_index - max_attention_size)
                x = attention(
                    roped_query,
                    temp_k[:, window_start:local_end_index],
                    temp_v[:, window_start:local_end_index],
                )

        # output
        x = x.flatten(2)
        x = self.o(x)

        # Return both output and cache update info
        if kv_cache is not None:
            return x, (current_end, local_end_index, cache_update_info)
        else:
            return x


class CausalWanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        self_attn_result = self.self_attn(
            (
                self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * (1 + e[1])
                + e[0]
            ).flatten(1, 2),
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            sink_recache_after_switch,
        )

        if kv_cache is not None:
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(
            1, 2
        )

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
            )
            y = self.ffn(
                (
                    self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                    * (1 + e[4])
                    + e[3]
                ).flatten(1, 2)
            )
            x = x + (
                y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]
            ).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)

        if cache_update_info is not None:
            # cache_update_info is already in the format (current_end, local_end_index, cache_update_info)
            return x, cache_update_info
        else:
            return x


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1])
            + e[0]
        )
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan2.2-TI2V-5B diffusion backbone with causal (streaming) attention.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="ti2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=48,
        dim=3072,
        ffn_dim=14336,
        freq_dim=256,
        text_dim=4096,
        out_dim=48,
        num_heads=24,
        num_layers=30,
        local_attn_size=-1,
        sink_size=0,
        num_frame_per_block=1,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 'ti2v'):
                Model variant - 't2v', 'i2v', or 'ti2v'. For 5B this is 'ti2v',
                which shares the text-only cross-attention path with 't2v'.
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 48):
                Input video channels (the Wan2.2 VAE has 48 latent channels)
            dim (`int`, *optional*, defaults to 3072):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 14336):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings (umt5-xxl)
            out_dim (`int`, *optional*, defaults to 48):
                Output video channels (Wan2.2 VAE latent channels)
            num_heads (`int`, *optional*, defaults to 24):
                Number of attention heads (head_dim = 3072 / 24 = 128)
            num_layers (`int`, *optional*, defaults to 30):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks. ti2v uses the text-only cross-attention path (same as t2v).
        cross_attn_type = "i2v_cross_attn" if model_type == "i2v" else "t2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        # head_dim=128: d//6=21 -> temporal=44, spatial_h=42, spatial_w=42 (sums
        # to 128). Same parametric split as the 1.3B model; verified against
        # upstream wan_5b/modules/model.py CausalWanModel.__init__.
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
                tmp + frame_seqlen * num_frame_per_block
            )

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return (
                    (kv_idx < ends[q_idx])
                    & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
                ) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block=4,
        local_attn_size=-1,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
                tmp + frame_seqlen * num_frame_per_block
            )

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return (
                    (kv_idx < ends[q_idx])
                    & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
                ) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        return block_mask

    def _apply_cache_updates(self, kv_cache, cache_update_infos, **kwargs):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
            **kwargs: Additional arguments (ignored, for compatibility with wrapper)
        """
        for block_index, (
            current_end,
            local_end_index,
            update_info,
        ) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]

                if update_info["action"] == "roll_and_insert":
                    # Apply rolling update
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get(
                        "write_start_index", local_start_index
                    )
                    write_end_index = update_info.get(
                        "write_end_index", local_end_index
                    )
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]

                    # Perform the rolling operation
                    cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
                        cache["k"][
                            :,
                            sink_tokens + num_evicted_tokens : sink_tokens
                            + num_evicted_tokens
                            + num_rolled_tokens,
                        ].clone()
                    )
                    cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
                        cache["v"][
                            :,
                            sink_tokens + num_evicted_tokens : sink_tokens
                            + num_evicted_tokens
                            + num_rolled_tokens,
                        ].clone()
                    )

                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (
                        write_end_index - write_start_index
                    ):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v

                elif update_info["action"] == "direct_insert":
                    # Direct insert
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get(
                        "write_start_index", local_start_index
                    )
                    write_end_index = update_info.get(
                        "write_end_index", local_end_index
                    )
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]

                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (
                        write_end_index - write_start_index
                    ):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v

            # Update indices: do not roll back pointers during recomputation
            is_recompute = (
                False if update_info is None else update_info.get("is_recompute", False)
            )
            if not is_recompute:
                kv_cache[block_index]["global_end_index"].fill_(current_end)
                kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        sink_recache_after_switch=False,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one chunk at a time.

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B] or [B, F]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            Tensor:
                Stacked denoised video tensors with original input shapes
        """

        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=False)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = (
            self.time_projection(e)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t.shape)
        )

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            sink_recache_after_switch=sink_recache_after_switch,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        cache_update_info = None
        cache_update_infos = []  # Collect cache update info for all blocks
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    cache_update_info = block_cache_update_info[:2]
                else:
                    x = result
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                result = block(x, **kwargs)
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    cache_update_info = block_cache_update_info[:2]
                else:
                    x = result

        # After all blocks are processed, apply cache updates in a single pass
        if kv_cache is not None and cache_update_infos:
            self._apply_cache_updates(kv_cache, cache_update_infos)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Non-streaming forward (teacher-forcing / full-sequence). Not used by the
        Scope inference pipeline, which always passes a kv_cache. Kept as a
        documented entry point; porting the training path is out of scope for
        the inference release.
        """
        raise NotImplementedError(
            "wan2_2.CausalWanModel training forward is not ported; inference uses "
            "_forward_inference with a kv_cache."
        )

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H, W]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist(), strict=False):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size, strict=False)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
