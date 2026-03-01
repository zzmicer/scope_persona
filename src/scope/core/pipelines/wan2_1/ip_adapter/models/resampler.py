# Perceiver Resampler for IP-Adapter face identity encoding.
# Ported from ByteDance Lynx: https://github.com/bytedance/lynx
#
# Takes ArcFace embedding [B, 1, 512] → outputs identity tokens [B, 16, output_dim]
# that get injected into Wan2.1 cross-attention layers.


import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads
        self.heads = heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape
        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = q.view(b, -1, h, q.shape[-1] // h).transpose(1, 2)
        k = k.view(b, -1, h, k.shape[-1] // h).transpose(1, 2)
        v = v.view(b, -1, h, v.shape[-1] // h).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, seq_len, -1)
        return self.to_out(out)


class Resampler(nn.Module):
    """Perceiver Resampler that compresses identity embeddings into fixed-size token set.

    Lynx Lite config: depth=4, dim=1280, dim_head=64, heads=20,
                      num_queries=16, embedding_dim=512, output_dim=2048
    """

    def __init__(
        self,
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=16,
        embedding_dim=512,
        output_dim=2048,
        ff_mult=4,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        """
        Args:
            x: [B, N, embedding_dim] — face embeddings (N=1 for single face)

        Returns:
            [B, num_queries, output_dim] — identity tokens for cross-attention
        """
        latents = self.latents.expand(x.size(0), -1, -1)
        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)
