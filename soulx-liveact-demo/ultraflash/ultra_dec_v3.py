"""
The released ultra-decoder-v3 weights do not load into UltraFlash's own
UltraDecoder/TAEHV classes: the checkpoint is a bare nn.Sequential saved as
`net.*`, and its layer list differs from theirs by two modules.

Reconstructed from the checkpoint's key/shape signature (21 modules, idx 0-20):
theirs, minus the leading Clamp(), minus the conv(n_f[2], n_f[3]) that sits
before the final ReLU. Every shape then matches exactly, so the weights load
strict=True.

3x PixelShuffleUp = 8x spatial and TGrow strides 1,2,2 = 4x temporal, which is
the Wan2.1 VAE's (4,8,8) stride -- confirming the reconstruction is the right one.
"""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/uflash/UltraFlash/inference")

from sr.ultra_decoder import (  # noqa: E402
    MemBlock,
    PixelShuffleUp,
    TGrow,
    apply_model_with_memblocks,
    conv,
)


class UltraDecoderV3(nn.Module):
    latent_channels = 16
    image_channels = 3

    def __init__(self, checkpoint_path=None):
        super().__init__()
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 3  # 2**2 - 1, from the two stride-2 TGrows
        self.net = nn.Sequential(
            conv(self.latent_channels, n_f[0]),                       # 0
            nn.ReLU(inplace=False),                                   # 1
            MemBlock(n_f[0], n_f[0]),                                 # 2
            MemBlock(n_f[0], n_f[0]),                                 # 3
            MemBlock(n_f[0], n_f[0]),                                 # 4
            PixelShuffleUp(n_f[0], n_f[0]),                           # 5
            TGrow(n_f[0], 1),                                         # 6
            conv(n_f[0], n_f[1], bias=False),                         # 7
            MemBlock(n_f[1], n_f[1]),                                 # 8
            MemBlock(n_f[1], n_f[1]),                                 # 9
            MemBlock(n_f[1], n_f[1]),                                 # 10
            PixelShuffleUp(n_f[1], n_f[1]),                           # 11
            TGrow(n_f[1], 2),                                         # 12
            conv(n_f[1], n_f[2], bias=False),                         # 13
            MemBlock(n_f[2], n_f[2]),                                 # 14
            MemBlock(n_f[2], n_f[2]),                                 # 15
            MemBlock(n_f[2], n_f[2]),                                 # 16
            PixelShuffleUp(n_f[2], n_f[2]),                           # 17
            TGrow(n_f[2], 2),                                         # 18
            nn.ReLU(inplace=False),                                   # 19
            conv(n_f[3], self.image_channels),                        # 20
        )
        if checkpoint_path is not None:
            sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            sd = sd.get("model", sd)
            # v3 names PixelShuffleUp's inner conv `block`; this repo's class calls it `up`.
            sd = {k.replace(".block.", ".up."): v for k, v in sd.items()}
            self.load_state_dict(sd, strict=True)

    def decode_video(self, x_btchw, parallel=False, show_progress_bar=False):
        x = apply_model_with_memblocks(self.net, x_btchw, parallel, show_progress_bar)
        if self.frames_to_trim > 0 and x.shape[1] > self.frames_to_trim:
            x = x[:, self.frames_to_trim:]
        return x

    @torch.no_grad()
    def decode_latents(self, latents_bcthw, parallel=False):
        """BCTHW latent -> BCTHW pixels in [-1, 1], matching WanVAE.decode's range.

        This net already emits [-1,1]; do NOT apply the `v*2-1` that UltraFlash's
        own WanUltraDecoder does. Fitted against a Wan VAE decode of the same
        latents, as-is scores MAE 5.2/255 while v*2-1 scores 67.1 (crushed, red
        cast), and the best-fit affine comes out at pixel = 122.6*net + 125.4,
        i.e. the plain [-1,1] -> [0,255] mapping.
        """
        dev = next(self.parameters()).device
        dt = next(self.parameters()).dtype
        x = latents_bcthw.permute(0, 2, 1, 3, 4).contiguous().to(device=dev, dtype=dt)
        v = self.decode_video(x, parallel=parallel).to(torch.float32)
        return v.permute(0, 2, 1, 3, 4).contiguous()


if __name__ == "__main__":
    import time

    ck = "/workspace/uflash/ckpt/v1.1-ultra-decoder-v3-ema_decoder.pth"
    m = UltraDecoderV3(ck).to("cuda:0", torch.float16).eval().requires_grad_(False)
    print("loaded strict=True OK")

    for (t, h, w, tag) in ((8, 104, 180, "HR 1440x832"), (8, 52, 90, "LR 720x416")):
        lat = torch.randn(1, 16, t, h, w, device="cuda:0", dtype=torch.float16)
        with torch.no_grad():
            for i in range(4):
                if i == 1:
                    torch.cuda.synchronize()
                    t0 = time.time()
                out = m.decode_latents(lat)
            torch.cuda.synchronize()
        dt = (time.time() - t0) / 3
        print(f"{tag}: {tuple(out.shape)}  {dt*1000:.1f} ms  "
              f"(range {out.min().item():.2f}..{out.max().item():.2f})")
