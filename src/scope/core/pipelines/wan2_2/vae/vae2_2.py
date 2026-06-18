# SPDX-License-Identifier: Apache-2.0
# Modified from NVlabs/LongLive wan_5b/modules/vae2_2.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# Wan2.2 VAE wrapper (48-channel latents, 16x spatial / 4x temporal).
#
# The public interface (encode_to_latent / decode_to_pixel / clear_cache /
# create_encoder_cache, plus the [B, F, C, H, W] latent layout and scale =
# [mean, 1/std] normalization) is kept drop-in compatible with the Wan2.1
# wrapper at ../../wan2_1/vae/wan.py (WanVAEWrapper), so the LongLive pipeline
# can use this VAE the same way it uses the 2.1 VAE. Only the underlying module
# (modules/vae.py) and the 48-element normalization constants differ.

import os

import torch

from .constants import WAN2_2_VAE_LATENT_MEAN, WAN2_2_VAE_LATENT_STD
from .modules.vae import _video_vae, count_conv3d

# Default checkpoint filename (Wan-AI/Wan2.2-TI2V-5B).
DEFAULT_VAE_FILENAME = "Wan2.2_VAE.pth"

# Optional LightVAE-5B variant (Skywork/Matrix-Game-3.0, MG-LightVAE.pth).
LIGHTVAE_FILENAME = "MG-LightVAE.pth"

# Compression factors (Wan2.2 vae_stride = (4, 16, 16)).
VAE_TEMPORAL_DOWNSAMPLE_FACTOR = 4
VAE_SPATIAL_DOWNSAMPLE_FACTOR = 16
VAE_Z_DIM = 48

# Architecture config (from upstream wan_5b/modules/vae2_2.py).
VAE_DIM = 160
VAE_DEC_DIM = 256
VAE_DIM_MULT = [1, 2, 4, 4]
VAE_TEMPERAL_DOWNSAMPLE = [False, True, True]


class Wan2_2_VAEWrapper(torch.nn.Module):
    """Unified VAE wrapper for Wan2.2 models.

    Supports streaming (cached) and batch encoding/decoding. Normalization is
    always applied during encoding for consistent latent distributions.

    The 16x spatial compression of Wan2.2 (vs 8x in Wan2.1) comes from a
    ``patchify(patch_size=2)`` step on encode (and ``unpatchify`` on decode) on
    top of the 8x conv downsample, combined with the ``Down_ResidualBlock`` /
    ``Up_ResidualBlock`` residual-shortcut stages. Latents have ``z_dim=48``.

    Args:
        model_dir: Base directory containing model files.
        model_name: Model subdirectory name (e.g., "Wan2.2-TI2V-5B").
        vae_path: Explicit path to the VAE checkpoint (overrides
            model_dir/model_name).
        use_lightvae: If True, load the LightVAE-5B variant
            (Skywork/Matrix-Game-3.0, MG-LightVAE.pth) instead of the full
            Wan2.2 VAE. The full Wan2.2 VAE is the default.
    """

    def __init__(
        self,
        model_dir: str = "wan_models",
        model_name: str = "Wan2.2-TI2V-5B",
        vae_path: str | None = None,
        use_lightvae: bool = False,
    ):
        super().__init__()

        # Determine paths: explicit vae_path > model_dir/model_name default.
        if vae_path is None:
            filename = LIGHTVAE_FILENAME if use_lightvae else DEFAULT_VAE_FILENAME
            vae_path = os.path.join(model_dir, model_name, filename)
        self.vae_path = vae_path
        self.use_lightvae = use_lightvae

        self.register_buffer(
            "mean", torch.tensor(WAN2_2_VAE_LATENT_MEAN, dtype=torch.float32)
        )
        self.register_buffer(
            "std", torch.tensor(WAN2_2_VAE_LATENT_STD, dtype=torch.float32)
        )
        self.z_dim = VAE_Z_DIM
        self.spatial_downsample_factor = VAE_SPATIAL_DOWNSAMPLE_FACTOR
        self.temporal_downsample_factor = VAE_TEMPORAL_DOWNSAMPLE_FACTOR

        self.model = (
            _video_vae(
                pretrained_path=vae_path,
                z_dim=self.z_dim,
                dim=VAE_DIM,
                dec_dim=VAE_DEC_DIM,
                dim_mult=VAE_DIM_MULT,
                temperal_downsample=VAE_TEMPERAL_DOWNSAMPLE,
            )
            .eval()
            .requires_grad_(False)
        )

        # Cache encoder conv count for dynamic cache sizing.
        self._encoder_conv_count = count_conv3d(self.model.encoder)

    def _get_scale(self, device: torch.device, dtype: torch.dtype) -> list:
        """Get normalization scale parameters on the correct device/dtype."""
        return [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

    def _apply_encoding_normalization(
        self, latent: torch.Tensor, scale: list
    ) -> torch.Tensor:
        """Apply normalization to encoded latents."""
        if isinstance(scale[0], torch.Tensor):
            return (latent - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        return (latent - scale[0]) * scale[1]

    def _create_encoder_cache(self) -> list:
        """Create a fresh encoder feature cache with dynamic sizing."""
        return [None] * self._encoder_conv_count

    def create_encoder_cache(self):
        """Create encoder cache (interface parity with the Wan2.1 wrapper).

        Wan2.2's CausalConv3d uses raw frame prepending for temporal context,
        so it doesn't have TAE's MemBlock memory pollution issue. This method
        exists for interface compatibility with TAEWrapper / WanVAEWrapper.

        Returns:
            None (this VAE doesn't need an explicit external cache).
        """
        return None

    def encode_to_latent(
        self,
        pixel: torch.Tensor,
        use_cache: bool = True,
        encoder_cache=None,
    ) -> torch.Tensor:
        """Encode video pixels to latents.

        Args:
            pixel: Input video tensor [batch, channels, frames, height, width].
            use_cache: If True, use streaming encode (maintains cache state).
                If False, use batch encode with a temporary cache.
            encoder_cache: Ignored (interface parity with TAEWrapper).

        Returns:
            Latent tensor [batch, frames, channels, height, width].
        """
        device, dtype = pixel.device, pixel.dtype
        scale = self._get_scale(device, dtype)

        if use_cache:
            # Streaming encode - cache is maintained across calls.
            latent = self.model.stream_encode(pixel)
            # stream_encode returns unnormalized mu; normalize here.
            latent = self._apply_encoding_normalization(latent, scale)
        else:
            # Batch encode with a one-time cache (does not affect stream state).
            latent = self._encode_with_cache(pixel, scale, self._create_encoder_cache())

        # [batch, channels, frames, h, w] -> [batch, frames, channels, h, w]
        return latent.permute(0, 2, 1, 3, 4)

    def _encode_with_cache(
        self, x: torch.Tensor, scale: list, feat_cache: list
    ) -> torch.Tensor:
        """Encode using an explicit cache without affecting streaming state.

        This allows one-time encodes (e.g. first-frame re-encoding) without
        clearing the streaming cache. Mirrors the Wan2.1 wrapper, but applies
        ``patchify(patch_size=2)`` first (Wan2.2's extra spatial factor).
        """
        from .modules.vae import patchify

        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        for i in range(iter_):
            conv_idx = [0]
            if i == 0:
                out = self.model.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=feat_cache,
                    feat_idx=conv_idx,
                )
            else:
                out_ = self.model.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=feat_cache,
                    feat_idx=conv_idx,
                )
                out = torch.cat([out, out_], 2)

        mu, _ = self.model.conv1(out).chunk(2, dim=1)
        return self._apply_encoding_normalization(mu, scale)

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = True
    ) -> torch.Tensor:
        """Decode latents to video pixels.

        Args:
            latent: Latent tensor [batch, frames, channels, height, width].
            use_cache: If True, use streaming decode (maintains cache state).
                If False, use batch decode (clears cache before/after).

        Returns:
            Video tensor [batch, frames, channels, height, width] in [-1, 1].
        """
        # [batch, frames, channels, h, w] -> [batch, channels, frames, h, w]
        zs = latent.permute(0, 2, 1, 3, 4)

        # Decode on the VAE's own device (CUDA on Blackwell; the latent's device on
        # CPU/macOS) rather than hardcoding "cuda", so the bf16 CPU fallback path
        # the longlive2 pipeline advertises actually works.
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = latent.device
        zs = zs.to(device=model_device, dtype=torch.bfloat16)
        scale = self._get_scale(model_device, torch.bfloat16)

        if use_cache:
            output = self.model.stream_decode(zs, scale)
        else:
            output = self.model.decode(zs, scale)

        output = output.float().clamp_(-1, 1)
        # [batch, channels, frames, h, w] -> [batch, frames, channels, h, w]
        return output.permute(0, 2, 1, 3, 4)

    def clear_cache(self):
        """Clear encoder/decoder cache for the next sequence."""
        self.model.first_batch = True
