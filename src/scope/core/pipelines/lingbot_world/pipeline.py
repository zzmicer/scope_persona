"""LingBot-World-V2 scope pipeline: real-time interactive world streaming.

Wraps :class:`LingbotWorldSession` (persistent-KV-cache chunked generation)
behind the scope ``Pipeline`` interface so the standard server stack (WebRTC
streaming, frontend controller capture, prompt updates) drives it:

- each ``__call__`` generates one chunk (``chunk_size`` latent frames = 4x
  video frames) continuing the rolling session;
- ``ctrl_input`` (WASD/arrows/Q/E/mouse, captured by the frontend) becomes the
  camera trajectory for the chunk;
- prompt changes re-condition the model mid-stream (world events / character
  actions, e.g. "she waves at the camera");
- ``first_frame_image`` (UI image picker) or ``start_image`` (config) seeds the
  world; when the image-conditioning horizon runs out the session transparently
  re-seeds from the last generated frame.

The upstream ``wan`` package is imported from ``config.lingbot_repo`` at
construction (not vendored: CC BY-NC-SA 4.0). Runs on H100/H200-class GPUs.
"""

import logging
import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import torch

from scope.core.config import get_models_dir

from ..defaults import prepare_for_mode, resolve_input_mode
from ..interface import Pipeline, Requirements
from ..utils import validate_resolution
from .schema import LingbotWorldConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class LingbotWorldPipeline(Pipeline):
    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return LingbotWorldConfig

    def __init__(
        self,
        config=None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **load_params,
    ):
        if config is None:
            field_names = set(LingbotWorldConfig.model_fields)
            config = LingbotWorldConfig(
                **{k: v for k, v in load_params.items() if k in field_names}
            )
        validate_resolution(height=config.height, width=config.width, scale_factor=16)

        self.config = config
        self.device = device or torch.device("cuda")

        repo = os.environ.get("LINGBOT_WORLD_REPO") or config.lingbot_repo
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        ckpt_dir = config.ckpt_dir or str(
            get_models_dir() / "lingbot-world-v2-14b-causal-fast"
        )

        # Upstream imports happen only here — hosts without the checkout/GPU can
        # still import this module via the registry.
        import wan
        from wan.configs import WAN_CONFIGS

        # Upstream's cross-attn calls flash_attention directly (hard assert);
        # everything else goes through attention(), which falls back to SDPA.
        # Without a flash-attn build for this torch, route the direct call
        # through the SDPA-capable wrapper too.
        try:
            import flash_attn  # noqa: F401
        except ModuleNotFoundError:
            from wan.modules import attention as _wan_attention
            from wan.modules import model_fast as _wan_model_fast

            _wan_model_fast.flash_attention = _wan_attention.attention
            logger.warning(
                "lingbot-world: flash-attn not installed; using PyTorch SDPA"
            )

        logger.info("lingbot-world: loading pipeline from %s", ckpt_dir)
        self._pipe = wan.WanI2VCausal(
            config=WAN_CONFIGS["i2v-A14B"],
            checkpoint_dir=ckpt_dir,
            device_id=self.device.index or 0,
            rank=0,
            init_on_cpu=False,
            local_attn_size=config.local_attn_size,
            sink_size=config.sink_size,
            infer_mode="causal_fast",
        )

        self._session = None
        self._prompt = ""
        self._seed = config.base_seed
        self.first_call = True

    # ------------------------------------------------------------------
    def prepare(self, **kwargs) -> Requirements | None:
        return prepare_for_mode(self.__class__, self.config, kwargs)

    def __call__(self, **kwargs) -> dict:
        resolve_input_mode(kwargs)  # text-only; validates mode kwargs shape

        prompt = self._resolve_prompt(kwargs)
        image_path = kwargs.get("first_frame_image") or None

        # (Re)seed the world: first call, explicit new image, or cache reset.
        want_reset = bool(kwargs.get("init_cache", False)) and not self.first_call
        if self._session is None or image_path:
            if self._session is None and not (image_path or self.config.start_image):
                # No world yet: emit black frames until an image is picked in
                # the UI instead of erroring the stream loop.
                f = self.config.chunk_size * 4
                return {
                    "video": torch.zeros(f, self.config.height, self.config.width, 3)
                }
            self._start_session(image_path, prompt)
        elif want_reset:
            self._start_session(None, prompt, from_last_frame=True)
        self.first_call = False

        session = self._session
        if prompt and prompt != self._prompt:
            session.set_prompt(prompt)
            self._prompt = prompt

        # Horizon exhausted -> re-seed from the last generated frame so the
        # stream continues (KV cache restarts; world persists via the image).
        if session.latent_budget_left < self.config.chunk_size:
            logger.info("lingbot-world: horizon exhausted, re-seeding from last frame")
            self._start_session(None, self._prompt, from_last_frame=True)
            session = self._session

        from .actions import delta_from_ctrl

        delta = delta_from_ctrl(
            kwargs.get("ctrl_input"),
            move_speed=float(kwargs.get("move_speed", self.config.move_speed)),
            turn_deg=float(kwargs.get("turn_speed", self.config.turn_speed)),
        )
        track = []
        cur = session.cur_c2w.copy()
        for _ in range(self.config.chunk_size):
            cur = cur @ delta
            track.append(cur.copy())
        frames = session.step(np.stack(track))  # [3, f, h, w] in [-1, 1], cpu

        video = frames.permute(1, 2, 3, 0).add(1.0).div(2.0).clamp(0.0, 1.0)
        return {"video": video}

    # ------------------------------------------------------------------
    def _resolve_prompt(self, kwargs) -> str:
        raw = kwargs.get("prompts") or kwargs.get("prompt")
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else ""
        if isinstance(raw, dict):
            return raw.get("text", "")
        return raw or ""

    def _start_session(
        self, image_path: str | None, prompt: str, from_last_frame: bool = False
    ) -> None:
        from PIL import Image

        from .session import LingbotWorldSession

        if from_last_frame and self._session is not None:
            last = self._session.frames[-1][:, -1]  # [3, h, w] in [-1, 1]
            arr = ((last.permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
            img = Image.fromarray(arr)
            self._seed += 1
        else:
            path = image_path or self.config.start_image
            if not path:
                raise ValueError(
                    "lingbot-world needs a start image: set start_image in the "
                    "pipeline settings or pick a first-frame image in the UI"
                )
            img = Image.open(path).convert("RGB")

        self._prompt = prompt or self._prompt or "an explorable world"
        logger.info(
            "lingbot-world: new session (from_last_frame=%s, prompt=%r)",
            from_last_frame,
            self._prompt[:80],
        )
        self._session = LingbotWorldSession(
            self._pipe,
            img,
            self._prompt,
            max_frames=self.config.max_frames,
            chunk_size=self.config.chunk_size,
            max_area=self.config.height * self.config.width,
            seed=self._seed,
        )
