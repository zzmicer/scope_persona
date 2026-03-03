# Vendored from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/clip.py
# Modified to extract only the visual encoder portion needed for I2V conditioning.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from scope.core.pipelines.wan2_1.modules.attention import flash_attention

logger = logging.getLogger(__name__)

# CLIP image normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class CLIPLayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class CLIPSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim
        q, k, v = self.to_qkv(x).view(b, s, 3, n, d).unbind(2)
        p = self.attn_dropout if self.training else 0.0
        x = flash_attention(q, k, v, dropout_p=p, causal=False, version=2)
        x = x.reshape(b, s, c)
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)
        return x


class CLIPAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio,
        num_heads,
        post_norm=False,
        activation="gelu",
        attn_dropout=0.0,
        proj_dropout=0.0,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.post_norm = post_norm
        self.norm1 = CLIPLayerNorm(dim, eps=norm_eps)
        self.attn = CLIPSelfAttention(dim, num_heads, attn_dropout, proj_dropout)
        self.norm2 = CLIPLayerNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            QuickGELU() if activation == "quick_gelu" else nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(proj_dropout),
        )

    def forward(self, x):
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class CLIPVisionTransformer(nn.Module):
    """Vision Transformer from Wan2.1's CLIP (ViT-H/14, 1280-dim, 32 layers)."""

    def __init__(
        self,
        image_size=224,
        patch_size=14,
        dim=1280,
        mlp_ratio=4,
        out_dim=1024,
        num_heads=16,
        num_layers=32,
        pool_type="token",
        pre_norm=True,
        post_norm=False,
        activation="gelu",
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim

        # Embeddings
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=not pre_norm
        )
        if pool_type in ("token", "token_fc"):
            self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(
            gain
            * torch.randn(
                1,
                self.num_patches + (1 if pool_type in ("token", "token_fc") else 0),
                dim,
            )
        )
        self.dropout = nn.Dropout(embedding_dropout)

        # Transformer
        self.pre_norm = CLIPLayerNorm(dim, eps=norm_eps) if pre_norm else None
        self.transformer = nn.Sequential(
            *[
                CLIPAttentionBlock(
                    dim,
                    mlp_ratio,
                    num_heads,
                    post_norm,
                    activation,
                    attn_dropout,
                    proj_dropout,
                    norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_norm = CLIPLayerNorm(dim, eps=norm_eps)

    def forward(self, x, use_31_block=False):
        b = x.size(0)
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        x = self.dropout(x + self.pos_embedding)
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        if use_31_block:
            x = self.transformer[:-1](x)
            return x
        else:
            x = self.transformer(x)
            return x


class WanCLIPVisualEncoder(nn.Module):
    """CLIP visual encoder for Wan2.1 I2V conditioning.

    Extracts visual features from reference images using the ViT-H/14 vision
    encoder from Wan2.1's XLMRobertaCLIP model. Features are extracted from the
    first 31 transformer blocks (before final normalization) to provide rich
    spatial features for the I2V model's MLPProj cross-attention conditioning.

    Output shape: [B, 257, 1280] (1 CLS token + 256 patch tokens, 1280-dim features)
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.dtype = dtype

        # Create ViT-H/14 visual encoder with Wan2.1 CLIP config
        with torch.device("meta"):
            self.visual = CLIPVisionTransformer(
                image_size=224,
                patch_size=14,
                dim=1280,
                mlp_ratio=4,
                out_dim=1024,
                num_heads=16,
                num_layers=32,
                pool_type="token",
                pre_norm=True,
                post_norm=False,
                activation="gelu",
            )

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, device)
        else:
            self.visual = self.visual.to_empty(device=device)

        self.visual = self.visual.to(dtype=dtype)
        self.visual.eval()
        self.visual.requires_grad_(False)

        # Preprocessing: normalize to CLIP stats
        self.normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    def _load_checkpoint(self, checkpoint_path: str, device: torch.device | str):
        """Load visual encoder weights from the CLIP checkpoint."""
        logger.info(f"Loading CLIP visual encoder from {checkpoint_path}")

        # Handle both .pth and .safetensors formats
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_path, device="cpu")
        else:
            state_dict = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        # Extract only visual encoder weights (prefix: "visual.")
        # If already visual-only (no prefix), use as-is
        visual_state_dict = {}
        has_visual_prefix = any(k.startswith("visual.") for k in state_dict.keys())

        if has_visual_prefix:
            for k, v in state_dict.items():
                if k.startswith("visual."):
                    visual_state_dict[k.removeprefix("visual.")] = v
        else:
            # Already visual-only checkpoint
            visual_state_dict = state_dict

        if not visual_state_dict:
            raise ValueError(
                f"No visual encoder weights found in {checkpoint_path}. "
                "Expected keys starting with 'visual.' or visual-only checkpoint"
            )

        self.visual = self.visual.to_empty(device="cpu")
        self.visual.load_state_dict(visual_state_dict, strict=True)
        logger.info(
            f"Loaded {len(visual_state_dict)} visual encoder parameters from CLIP checkpoint"
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP visual features from images.

        Args:
            images: Image tensor in [-1, 1] range with shape [B, C, F, H, W]
                    (video latent format) or [B, C, H, W] (single image).
                    The first frame is used if video format is provided.

        Returns:
            CLIP visual features of shape [B, 257, 1280]
        """
        # Handle video format [B, C, F, H, W] -> take first frame
        if images.ndim == 5:
            images = images[:, :, 0]  # [B, C, H, W]

        # Scale from [-1, 1] to [0, 1]
        images = images.float().mul(0.5).add(0.5)

        # Resize to CLIP input size (224x224)
        images = F.interpolate(
            images,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
        )

        # Apply CLIP normalization
        images = self.normalize(images)

        # Extract features from first 31 blocks
        with torch.autocast(
            str(self.visual.patch_embedding.weight.device), dtype=self.dtype
        ):
            features = self.visual(images, use_31_block=True)

        return features
