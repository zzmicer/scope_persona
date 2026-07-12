"""Interactive LingBot-World-V2 session with a persistent KV cache.

Adapted from `wan.image2video.WanI2VCausal._generate_causal_fast` (upstream
lingbot-world-v2), restructured so generation happens turn-by-turn instead of
from one fixed action file:

- the self-attention KV cache, cross-attention cache, scheduler, and image
  conditioning live across turns;
- each turn supplies new camera poses (at the latent frame rate) and an
  optional new prompt (event), and produces the newly decoded video frames;
- VAE decode is windowed with a small latent overlap so per-turn decode cost
  stays constant (same approach verified for the OmniForcing streaming port).

Requires the upstream repo on sys.path (see demo.py / LINGBOT_WORLD_REPO).
"""

import logging
import math
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from wan.utils.cam_utils import get_plucker_embeddings

from .actions import default_intrinsics

logger = logging.getLogger(__name__)


class LingbotWorldSession:
    """One interactive world: an image, a rolling KV cache, and a camera."""

    def __init__(
        self,
        pipe,
        img,
        prompt: str,
        max_frames: int = 321,
        chunk_size: int = 4,
        max_area: int = 480 * 832,
        shift: float | None = None,
        timesteps_index: tuple[int, ...] = (0, 250, 500, 750),
        seed: int = 42,
        decode_overlap_latents: int = 2,
    ):
        assert pipe.infer_mode == "causal_fast"
        self.pipe = pipe
        self.chunk_size = chunk_size
        self.decode_overlap = decode_overlap_latents
        self.device = pipe.device
        cfg = pipe.config

        # ---- geometry -------------------------------------------------
        img_t = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        h, w = img_t.shape[1:]
        aspect_ratio = h / w
        self.lat_h = round(
            np.sqrt(max_area * aspect_ratio)
            // cfg.vae_stride[1]
            // cfg.patch_size[1]
            * cfg.patch_size[1]
        )
        self.lat_w = round(
            np.sqrt(max_area / aspect_ratio)
            // cfg.vae_stride[2]
            // cfg.patch_size[2]
            * cfg.patch_size[2]
        )
        self.h = self.lat_h * cfg.vae_stride[1]
        self.w = self.lat_w * cfg.vae_stride[2]

        lat_f_max = (max_frames - 1) // cfg.vae_stride[0] + 1
        self.lat_f_max = int(lat_f_max - (lat_f_max % chunk_size))
        f_max = (self.lat_f_max - 1) * 4 + 1

        self.frame_seqlen = (
            self.lat_h * self.lat_w // (cfg.patch_size[1] * cfg.patch_size[2])
        )
        self.max_seq_len = int(
            math.ceil(chunk_size * self.frame_seqlen / pipe.sp_size) * pipe.sp_size
        )

        # ---- image conditioning (whole horizon, computed once) --------
        msk = torch.ones(1, f_max, self.lat_h, self.lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, self.lat_h, self.lat_w)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat(
            [
                torch.nn.functional.interpolate(
                    img_t[None].cpu(), size=(self.h, self.w), mode="bicubic"
                ).transpose(0, 1),
                torch.zeros(3, f_max - 1, self.h, self.w),
            ],
            dim=1,
        ).to(self.device)
        y = pipe.vae.encode([vae_input])[0]
        self.y = torch.concat([msk, y])  # [20, lat_f_max, lat_h, lat_w]
        del vae_input

        # ---- text conditioning -----------------------------------------
        # Keep T5 resident; on 140GB-class GPUs there is no need to offload,
        # and event prompts re-encode mid-session.
        pipe.text_encoder.model.to(self.device)
        self.context = pipe.text_encoder([prompt], self.device)
        self._cross_refresh = True  # next forward must (re)fill the cross-attn cache

        # ---- scheduler / noise ----------------------------------------
        pipe.scheduler.set_timesteps(
            pipe.num_train_timesteps,
            shift=shift if shift is not None else cfg.sample_shift,
        )
        self.timesteps = pipe.scheduler.timesteps[list(timesteps_index)]
        self.seed_g = torch.Generator(device=self.device)
        self.seed_g.manual_seed(seed)

        # ---- persistent caches -----------------------------------------
        model_args = pipe.model.config
        if pipe.local_attn_size > -1:
            self.kv_size = self.frame_seqlen * pipe.local_attn_size
        else:
            self.kv_size = self.frame_seqlen * self.lat_f_max
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // pipe.sp_size
        self.self_kv_cache = pipe._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=[1, self.kv_size, local_num_heads, head_dim],
            dtype=pipe.pipe_dtype,
            device=self.device,
        )
        self.cross_kv_cache = pipe._initialize_crossattn_cache(
            num_layers=model_args.num_layers,
            shape=[1, 512, model_args.num_heads, head_dim],
            dtype=pipe.pipe_dtype,
            device=self.device,
        )

        # ---- rolling state ----------------------------------------------
        self.Ks = torch.from_numpy(default_intrinsics(self.w, self.h)).float()
        self.cur_c2w = np.eye(4)
        self.lat_next = 0  # latent frames generated so far
        self.latents: list[torch.Tensor] = []  # per-turn [16, n, lat_h, lat_w]
        self.frames: list[torch.Tensor] = []  # per-turn decoded [3, f, h, w] cpu

    # ------------------------------------------------------------------
    @property
    def frames_generated(self) -> int:
        return sum(f.shape[1] for f in self.frames)

    @property
    def latent_budget_left(self) -> int:
        return self.lat_f_max - self.lat_next

    def set_prompt(self, prompt: str) -> None:
        """Swap the event/scene prompt; takes effect from the next chunk."""
        self.context = self.pipe.text_encoder([prompt], self.device)
        self._cross_refresh = True

    # ------------------------------------------------------------------
    def step(self, poses_c2w: np.ndarray) -> torch.Tensor:
        """Generate video continuing the session along the given camera track.

        poses_c2w: [n, 4, 4] absolute camera-to-world poses at the latent
        frame rate, continuing from (not including) self.cur_c2w; n must be a
        multiple of chunk_size. Returns the newly decoded frames [3, f, h, w]
        (uint8-range float in [-1, 1]) and appends them to self.frames.
        """
        n_lat = len(poses_c2w)
        assert n_lat % self.chunk_size == 0, "pose track must fill whole chunks"
        if n_lat > self.latent_budget_left:
            raise RuntimeError(
                f"latent budget exhausted: requested {n_lat}, left {self.latent_budget_left}"
            )

        pipe = self.pipe
        t0 = time.perf_counter()

        # framewise deltas, bridging from the current pose
        track = np.concatenate([self.cur_c2w[None], poses_c2w], axis=0)
        deltas = np.matmul(np.linalg.inv(track[:-1]), track[1:])
        deltas_t = torch.from_numpy(deltas).float().to(self.device)

        Ks = self.Ks.repeat(n_lat, 1).to(self.device)
        plucker = get_plucker_embeddings(deltas_t, Ks, self.h, self.w)
        plucker = rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=self.h // self.lat_h,
            c2=self.w // self.lat_w,
        )[None]
        plucker = rearrange(
            plucker,
            "b (f h w) c -> b c f h w",
            f=n_lat,
            h=self.lat_h,
            w=self.lat_w,
        ).to(pipe.param_dtype)

        noise = torch.randn(
            16,
            n_lat,
            self.lat_h,
            self.lat_w,
            dtype=torch.float32,
            generator=self.seed_g,
            device=self.device,
        )

        y_turn = self.y[:, self.lat_next : self.lat_next + n_lat]

        new_latents = []
        with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
            for ci in range(n_lat // self.chunk_size):
                sl = slice(ci * self.chunk_size, (ci + 1) * self.chunk_size)
                current_latent = noise[:, sl]
                chunk_id_global = self.lat_next // self.chunk_size + ci

                kwargs = {
                    "context": [self.context[0]],
                    "seq_len": self.max_seq_len,
                    "y": [y_turn[:, sl]],
                    "dit_cond_dict": {
                        "c2ws_plucker_emb": plucker[:, :, sl].chunk(1, dim=0)
                    },
                    "kv_cache": self.self_kv_cache,
                    "crossattn_cache": self.cross_kv_cache,
                    "current_start": chunk_id_global
                    * self.chunk_size
                    * self.frame_seqlen,
                    "max_attention_size": self.kv_size,
                    "frame_seqlen": self.frame_seqlen,
                }

                for ti in range(len(self.timesteps)):
                    timestep = torch.stack([self.timesteps[ti]]).to(self.device)
                    noise_pred = pipe.model(
                        x=[current_latent],
                        t=timestep,
                        cross_attn_first_call=self._cross_refresh,
                        **kwargs,
                    )[0]
                    self._cross_refresh = False
                    x0 = pipe._convert_flow_pred_to_x0(
                        flow_pred=noise_pred,
                        xt=current_latent,
                        timestep=self.timesteps[ti],
                        scheduler=pipe.scheduler,
                    )
                    if ti < len(self.timesteps) - 1:
                        current_latent = pipe.scheduler.add_noise(
                            x0,
                            torch.randn(
                                x0.shape,
                                generator=self.seed_g,
                                device=x0.device,
                                dtype=x0.dtype,
                            ),
                            self.timesteps[ti + 1],
                        )

                new_latents.append(x0)

                # write the clean chunk into the KV cache
                timestep0 = torch.stack([self.timesteps[-1] * 0.0]).to(self.device)
                pipe.model(x=[x0], t=timestep0, cross_attn_first_call=False, **kwargs)

            gen_s = time.perf_counter() - t0
            new_latents = torch.cat(new_latents, dim=1)

            # ---- windowed decode ----------------------------------------
            prev = self.latents[-1] if self.latents else None
            if prev is not None:
                ctx = min(self.decode_overlap, prev.shape[1])
                window = torch.cat([prev[:, -ctx:], new_latents], dim=1)
            else:
                ctx = 0
                window = new_latents
            decoded = pipe.vae.decode([window])[0]
            if ctx > 0:
                decoded = decoded[:, -n_lat * 4 :]
            # first turn: (n_lat-1)*4+1 frames, all new

        self.latents.append(new_latents)
        self.frames.append(decoded.cpu())
        self.cur_c2w = poses_c2w[-1].copy()
        self.lat_next += n_lat

        total_s = time.perf_counter() - t0
        logger.info(
            "turn: %d latent frames -> %d video frames in %.1fs (gen %.1fs, decode %.1fs)",
            n_lat,
            decoded.shape[1],
            total_s,
            gen_s,
            total_s - gen_s,
        )
        return decoded

    # ------------------------------------------------------------------
    def video(self) -> torch.Tensor:
        """All frames so far, [3, F, h, w] in [-1, 1]."""
        return torch.cat(self.frames, dim=1)

    def save(self, path: str, fps: int = 16) -> str:
        import imageio

        vid = self.video()
        arr = ((vid.clamp(-1, 1).permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).astype(
            np.uint8
        )
        imageio.mimwrite(path, list(arr), fps=fps, quality=8)
        return path
