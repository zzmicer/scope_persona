# Modified from https://github.com/ali-vilab/VACE/blob/48eb44f1c4be87cc65a98bff985a26976841e9f3/vace/models/wan/modules/model.py
# Adapted for causal/autoregressive generation with factory pattern
# Pipeline-agnostic using duck typing - works with any CausalWanModel
import inspect
import math

import torch
import torch.nn as nn

from .attention_blocks import (
    create_base_attention_block_class,
    create_vace_attention_block_class,
)


# TODO: Consolidate this with other pipeline implementations into a shared wan2_1/utils module.
# This is a standard sinusoidal positional embedding - identical across all pipelines apart from krea which has forced dtype
def sinusoidal_embedding_1d(dim, position):
    """
    Standard sinusoidal positional embedding.

    Args:
        dim: Embedding dimension
        position: Position tensor of shape [B]

    Returns:
        Embeddings of shape [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=position.device) / half
    )
    args = position[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class CausalVaceWanModel(nn.Module):
    """
    VACE wrapper that adds reference image conditioning to any CausalWanModel.

    Uses composition to wrap an existing CausalWanModel instance.
    Pipeline-agnostic via duck typing - works with longlive, streamdiffusionv2,
    krea_realtime_video, reward_forcing, or any future CausalWanModel implementation.
    """

    def __init__(
        self,
        causal_wan_model,
        vace_in_dim=96,
        vace_layers=None,
    ):
        super().__init__()

        # Store wrapped model
        self.causal_wan_model = causal_wan_model

        # Extract configuration from wrapped model via duck typing
        self.num_layers = causal_wan_model.num_layers
        self.dim = causal_wan_model.dim
        self.ffn_dim = causal_wan_model.ffn_dim
        self.num_heads = causal_wan_model.num_heads
        self.qk_norm = causal_wan_model.qk_norm
        self.cross_attn_norm = causal_wan_model.cross_attn_norm
        self.eps = causal_wan_model.eps
        self.patch_size = causal_wan_model.patch_size
        self.in_dim = causal_wan_model.in_dim

        # Pipeline-specific attributes (duck typed with defaults)
        self.local_attn_size = getattr(causal_wan_model, "local_attn_size", -1)
        self.window_size = getattr(causal_wan_model, "window_size", (-1, -1))
        if hasattr(causal_wan_model, "config") and hasattr(
            causal_wan_model.config, "sink_size"
        ):
            self.sink_size = causal_wan_model.config.sink_size
        else:
            self.sink_size = getattr(causal_wan_model, "sink_size", 0)

        # VACE configuration
        self.vace_layers = (
            list(range(0, self.num_layers, 2)) if vace_layers is None else vace_layers
        )
        self.vace_in_dim = vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # Get the original block class BEFORE replacing blocks
        self._original_block_class = type(causal_wan_model.blocks[0])

        # Create factory-generated classes for this pipeline's block type
        self._BaseWanAttentionBlock = create_base_attention_block_class(
            self._original_block_class
        )
        self._VaceWanAttentionBlock = create_vace_attention_block_class(
            self._original_block_class
        )

        # Replace blocks with hint-injection-enabled versions
        self._replace_blocks_with_hint_injection_support()

        # Create VACE blocks (parallel processing path for reference images)
        # Created on CPU by default for memory efficiency
        self._create_vace_blocks()

        # VACE patch embedding - create on CPU for memory efficiency
        # Will be populated with weights via load_vace_weights_only()
        with torch.device("cpu"):
            self.vace_patch_embedding = nn.Conv3d(
                self.vace_in_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )

        # Cache block forward signature for dynamic parameter filtering
        # This allows the VACE model to work with any CausalWanModel implementation
        self._block_forward_params = self._get_block_forward_params()

        # Detect whether wrapped model uses new-style (freqs_cos/sin) or old-style (grid_sizes/freqs) interface
        parent_sig = inspect.signature(self._original_block_class.forward)
        self._uses_new_style = "freqs_cos" in parent_sig.parameters

    def _get_block_init_kwargs(self):
        """Get initialization kwargs for creating new blocks.

        Uses duck typing to determine which parameters the block class expects.
        """
        cross_attn_type = "t2v_cross_attn"

        # Base kwargs that all blocks should have
        kwargs = {
            "cross_attn_type": cross_attn_type,
            "dim": self.dim,
            "ffn_dim": self.ffn_dim,
            "num_heads": self.num_heads,
            "qk_norm": self.qk_norm,
            "cross_attn_norm": self.cross_attn_norm,
            "eps": self.eps,
        }

        # Add pipeline-specific kwargs based on what the original block class expects
        sig = inspect.signature(self._original_block_class.__init__)
        params = sig.parameters

        if "local_attn_size" in params:
            kwargs["local_attn_size"] = self.local_attn_size
        if "sink_size" in params:
            kwargs["sink_size"] = self.sink_size
        if "window_size" in params:
            kwargs["window_size"] = self.window_size

        return kwargs

    def _get_block_forward_params(self):
        """Get the set of parameter names accepted by the block's forward method.

        Inspects the original block class's forward signature to determine which
        parameters should be passed through to blocks. This allows the VACE model
        to work with any CausalWanModel implementation without hardcoding parameter names.

        Returns:
            set: Parameter names accepted by block.forward(), or None if the block
                 accepts **kwargs (VAR_KEYWORD) and can handle any parameters.
        """
        sig = inspect.signature(self._original_block_class.forward)

        # If block accepts **kwargs, return None to indicate all params are accepted
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_keyword:
            return None

        return set(sig.parameters.keys())

    def _filter_block_kwargs(self, block_kwargs, block_index):
        """Filter and prepare kwargs for a specific block.

        Handles two types of parameters:
        1. Per-block indexed: Lists with length matching num_blocks (e.g., kv_bank)
           These get indexed with block_index.
        2. Shared: Scalar/other values passed to all blocks as-is

        Only includes parameters that the block's forward method accepts.

        Args:
            block_kwargs: Dict of additional kwargs from _forward_inference
            block_index: Index of the current block

        Returns:
            Dict of kwargs filtered and prepared for this specific block
        """
        if not block_kwargs:
            return {}

        filtered = {}
        for key, value in block_kwargs.items():
            # Skip if block doesn't accept this parameter
            if (
                self._block_forward_params is not None
                and key not in self._block_forward_params
            ):
                continue

            # Check if this is a per-block indexed parameter (list matching block count)
            if isinstance(value, list | tuple) and len(value) == self.num_layers:
                filtered[key] = value[block_index]
            else:
                filtered[key] = value

        return filtered

    def _replace_blocks_with_hint_injection_support(self):
        """Replace blocks with BaseWanAttentionBlock to support hint injection.

        Creates new block instances of the factory-generated class and copies
        weights from the original blocks. Uses proper inheritance (not composition),
        so state_dict paths are preserved.

        Memory-optimized: replaces blocks one at a time to avoid doubling memory usage.
        """
        original_blocks = self.causal_wan_model.blocks

        # Get device and dtype from original blocks
        orig_dtype = next(original_blocks[0].parameters()).dtype
        orig_device = next(original_blocks[0].parameters()).device

        # Get initialization kwargs
        block_kwargs = self._get_block_init_kwargs()

        # Replace blocks ONE AT A TIME to minimize peak memory usage
        # This avoids having both old and new blocks in memory simultaneously
        new_blocks = nn.ModuleList()

        for i in range(self.num_layers):
            block_id = self.vace_layers_mapping[i] if i in self.vace_layers else None
            orig_block = original_blocks[i]

            # Create new block on CPU first to avoid GPU memory spike
            with torch.device("cpu"):
                new_block = self._BaseWanAttentionBlock(
                    **block_kwargs,
                    block_id=block_id,
                )

            # Get state dict from original (on GPU) and copy to new block (on CPU)
            orig_state = orig_block.state_dict()
            new_state = new_block.state_dict()
            saved_block_id = new_block.block_id

            for key in orig_state.keys():
                if key in new_state:
                    # Copy to CPU (where new_block is)
                    new_state[key] = orig_state[key].to("cpu")

            new_block.load_state_dict(new_state, strict=False, assign=True)
            new_block.block_id = saved_block_id

            # Move original block to CPU first to free GPU memory immediately
            # This is safe because we've already copied the weights to CPU
            orig_block.to("cpu")

            # Now move new block to target device/dtype (reusing freed GPU memory)
            new_block = new_block.to(device=orig_device, dtype=orig_dtype)
            new_block.eval()

            new_blocks.append(new_block)

            # Clear CUDA cache periodically to help with fragmentation
            if i % 10 == 0:
                torch.cuda.empty_cache()

        # Final cache clear
        torch.cuda.empty_cache()

        # Replace blocks in wrapped model
        self.causal_wan_model.blocks = new_blocks

        # Also register blocks on self for LoRA compatibility
        self.blocks = new_blocks

    def _create_vace_blocks(self):
        """Create VACE blocks for parallel processing of reference images.

        Creates blocks on CPU by default to save GPU memory.
        They will be populated with weights later via load_vace_weights_only().
        """
        # Get dtype from existing blocks (but create on CPU for memory efficiency)
        orig_dtype = next(self.blocks[0].parameters()).dtype

        # Get initialization kwargs
        block_kwargs = self._get_block_init_kwargs()

        # Create VACE blocks on CPU to save GPU memory
        # They will receive weights via load_vace_weights_only() and can optionally
        # be moved to GPU later if memory permits
        vace_blocks = nn.ModuleList()
        with torch.device("cpu"):
            for block_id in range(len(self.vace_layers)):
                vace_block = self._VaceWanAttentionBlock(
                    **block_kwargs,
                    block_id=block_id,
                )
                vace_blocks.append(vace_block)

        # Keep on CPU, use orig_dtype
        vace_blocks.to(dtype=orig_dtype)

        self.vace_blocks = vace_blocks

    def _build_vace_block_kwargs(
        self, e0, context, context_lens, crossattn_cache, x, seq_len
    ):
        """Build kwargs for VACE block forward_vace calls based on wrapped model type."""
        if self._uses_new_style:
            # New-style: LongLive with precomputed RoPE (freqs_cos/sin)
            # VACE blocks don't use caching — pass empty cache tensors
            empty_k = torch.empty(
                1,
                0,
                self.num_heads,
                self.dim // self.num_heads,
                dtype=x.dtype,
                device=x.device,
            )
            return {
                "e": e0,
                "freqs_cos": None,  # Will be set per-call
                "freqs_sin": None,  # Will be set per-call
                "context": context,
                "cache_k": empty_k,
                "cache_v": empty_k,
                "crossattn_cache": None,
            }
        else:
            # Old-style: grid_sizes/freqs/block_mask
            return {
                "e": e0,
                "seq_lens": torch.tensor([x.size(1)], dtype=torch.long),
                "grid_sizes": None,  # Set by caller
                "freqs": self.causal_wan_model.freqs,
                "context": context,
                "context_lens": context_lens,
                "block_mask": getattr(self.causal_wan_model, "block_mask", None),
                "crossattn_cache": None,
            }

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        e0,
        context,
        context_lens,
        crossattn_cache,
        # Old-style params (may be None for new-style models)
        seq_lens=None,
        grid_sizes=None,
        freqs=None,
        block_mask=None,
        # New-style params (may be None for old-style models)
        freqs_cos=None,
        freqs_sin=None,
    ):
        """Process VACE context to generate hints."""
        # Get target dtype from vace_patch_embedding parameters
        target_dtype = next(self.vace_patch_embedding.parameters()).dtype

        # Convert all VACE context to model dtype first
        vace_context_converted = [u.to(dtype=target_dtype) for u in vace_context]

        # Embed VACE context
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context_converted]
        c = [u.flatten(2).transpose(1, 2) for u in c]

        # Pad to seq_len
        c = torch.cat(
            [
                torch.cat(
                    [u, u.new_zeros(1, max(0, seq_len - u.size(1)), u.size(2))], dim=1
                )
                for u in c
            ]
        )

        # Build block kwargs based on model type
        if self._uses_new_style:
            empty_k = torch.empty(
                1,
                0,
                self.num_heads,
                self.dim // self.num_heads,
                dtype=x.dtype,
                device=x.device,
            )
            block_kwargs = {
                "e": e0,
                "freqs_cos": freqs_cos,
                "freqs_sin": freqs_sin,
                "context": context,
                "cache_k": empty_k,
                "cache_v": empty_k,
            }
        else:
            block_kwargs = {
                "e": e0,
                "seq_lens": seq_lens,
                "grid_sizes": grid_sizes,
                "freqs": freqs,
                "context": context,
                "context_lens": context_lens,
                "block_mask": block_mask,
            }

        # Process through VACE blocks
        for _block_idx, block in enumerate(self.vace_blocks):
            c = block.forward_vace(c, x, **block_kwargs)

        # Extract hints
        hints = torch.unbind(c)[:-1]
        return hints

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len=None,
        clip_fea=None,
        y=None,
        vace_context=None,
        vace_context_scale=1.0,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        **block_kwargs,
    ):
        """Forward pass with optional VACE conditioning."""
        device = self.causal_wan_model.patch_embedding.weight.device
        if self.causal_wan_model.freqs.device != device:
            self.causal_wan_model.freqs = self.causal_wan_model.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=False)]

        # Embeddings
        x = [self.causal_wan_model.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]

        if self._uses_new_style:
            # New-style: single tensor, compute RoPE cos/sin
            x = torch.cat(x)
            f, h, w = grid_sizes[0].tolist()
            from scope.core.pipelines.longlive.modules.causal_model import (
                precompute_freqs_i,
            )

            start_frame = current_start // (h * w)
            freqs_cos, freqs_sin = precompute_freqs_i(
                self.causal_wan_model.freqs, f, h, w, start_frame
            )
        else:
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            if seq_len is not None:
                assert seq_lens.max() <= seq_len
            x = torch.cat(x)

        # Time embeddings
        e = self.causal_wan_model.time_embedding(
            sinusoidal_embedding_1d(
                self.causal_wan_model.freq_dim, t.flatten()
            ).type_as(x)
        )
        e0 = (
            self.causal_wan_model.time_projection(e)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t.shape)
        )

        # Context
        context_lens = None
        context = self.causal_wan_model.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [
                            u,
                            u.new_zeros(
                                self.causal_wan_model.text_len - u.size(0), u.size(1)
                            ),
                        ]
                    )
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.causal_wan_model.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        # Generate VACE hints
        hints = None
        if vace_context is not None:
            if self._uses_new_style:
                hints = self.forward_vace(
                    x,
                    vace_context,
                    seq_len if seq_len else f * h * w,
                    e0,
                    context,
                    context_lens,
                    crossattn_cache,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                )
            else:
                hints = self.forward_vace(
                    x,
                    vace_context,
                    seq_len,
                    e0,
                    context,
                    context_lens,
                    crossattn_cache,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    freqs=self.causal_wan_model.freqs,
                    block_mask=self.causal_wan_model.block_mask,
                )

        # Build base kwargs for transformer blocks
        if self._uses_new_style:
            base_kwargs = {
                "e": e0,
                "freqs_cos": freqs_cos,
                "freqs_sin": freqs_sin,
                "context": context,
                "hints": hints,
                "context_scale": vace_context_scale,
            }
        else:
            base_kwargs = {
                "e": e0,
                "seq_lens": seq_lens,
                "grid_sizes": grid_sizes,
                "freqs": self.causal_wan_model.freqs,
                "context": context,
                "context_lens": context_lens,
                "block_mask": self.causal_wan_model.block_mask,
                "hints": hints,
                "context_scale": vace_context_scale,
            }

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        # Process through blocks
        new_kvs = []
        for block_index, block in enumerate(self.blocks):
            # Build per-block kwargs
            filtered_block_kwargs = self._filter_block_kwargs(block_kwargs, block_index)

            if self._uses_new_style:
                # New-style: pass cache_k, cache_v tensors directly
                fill_level = self.causal_wan_model.fill_level
                if fill_level > 0:
                    cache_k = kv_cache[block_index]["k"][:, :fill_level]
                    cache_v = kv_cache[block_index]["v"][:, :fill_level]
                else:
                    empty = torch.empty(
                        1,
                        0,
                        self.num_heads,
                        self.dim // self.num_heads,
                        dtype=x.dtype,
                        device=x.device,
                    )
                    cache_k = empty
                    cache_v = empty

                per_block_kwargs = {
                    "cache_k": cache_k,
                    "cache_v": cache_v,
                    "crossattn_cache": crossattn_cache[block_index]
                    if crossattn_cache
                    else None,
                    **filtered_block_kwargs,
                }
            else:
                per_block_kwargs = {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index]
                    if crossattn_cache
                    else None,
                    "current_start": current_start,
                    **filtered_block_kwargs,
                }

            if torch.is_grad_enabled() and self.causal_wan_model.gradient_checkpointing:
                kwargs = {**base_kwargs, **per_block_kwargs}
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_info = result
                    new_kvs.append(block_cache_info)
                else:
                    x = result
            else:
                kwargs = {**base_kwargs, **per_block_kwargs}
                result = block(x, **kwargs)
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_info = result
                    new_kvs.append(block_cache_info)
                else:
                    x = result

        # Apply cache updates
        if kv_cache is not None and new_kvs:
            if self._uses_new_style:
                # New-style: use fill_level-based roll update
                self.causal_wan_model._roll_update_cache(kv_cache, new_kvs)
            else:
                # Old-style: use _apply_cache_updates
                cache_update_infos = [(i, info) for i, info in enumerate(new_kvs)]
                self.causal_wan_model._apply_cache_updates(
                    kv_cache, cache_update_infos, **block_kwargs
                )

        x = self.causal_wan_model.head(
            x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2)
        )

        if self._uses_new_style:
            x = self.causal_wan_model.unpatchify(x, f, h, w)
        else:
            x = self.causal_wan_model.unpatchify(x, grid_sizes)
            x = torch.stack(x)

        return x

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self.causal_wan_model._forward_train(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.causal_wan_model, name)
