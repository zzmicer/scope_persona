"""OmniForcing (LTX-2 causal AV) scope pipeline.

Streaming autoregressive **audio + video** generation, distilled from the
bidirectional LTX-2 teacher. Structurally analogous to ``longlive2`` (causal
KV-cache transformer + short distilled schedule), but on the LTX-2 family and
emitting an audio track alongside video.

All OmniForcing/LTX-specific construction + the AR inference loop are isolated in
``runtime.py`` (lazy, GPU-host-only imports) so this module — and therefore the
pipeline registry — imports cleanly on hosts without the LTX stack installed.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from diffusers.modular_pipelines import PipelineState
from omegaconf import OmegaConf

from scope.core.config import get_models_dir

from ..defaults import prepare_for_mode, resolve_input_mode
from ..interface import Pipeline, Requirements
from ..utils import Quantization, validate_resolution
from . import runtime
from .schema import OmniForcingConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class OmniForcingPipeline(Pipeline):
    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return OmniForcingConfig

    def __init__(
        self,
        config=None,
        quantization: Quantization | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **load_params,
    ):
        # Plugin load path: the pipeline manager constructs plugin pipelines as
        # ``pipeline_class(**merged_params)`` where ``merged_params`` are the config
        # field values (schema defaults overridden by UI load params), NOT a
        # ``config`` object. Build the config from those; the explicit ``config=``
        # path (tests / builtin-style construction) is also supported.
        if config is None:
            field_names = set(OmniForcingConfig.model_fields)
            config = OmniForcingConfig(
                **{k: v for k, v in load_params.items() if k in field_names}
            )

        # LTX-2 video VAE downsamples 32x spatially -> height/width multiples of 32.
        validate_resolution(
            height=config.height,
            width=config.width,
            scale_factor=32,
        )

        self.config = config
        self.dtype = dtype
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # The plugin path doesn't inject model_dir; resolve it from scope config
        # (DAYDREAM_SCOPE_MODELS_DIR / default), matching download_models.
        model_dir = getattr(config, "model_dir", None) or str(get_models_dir())
        self.model_config = OmegaConf.load(Path(__file__).parent / "model.yaml")

        # Build the GPU runtime (loads LTX-2 base + OmniForcing generator + VAEs +
        # vocoder + Gemma, wraps CausalAVInferencePipeline). On hosts without the
        # LTX stack installed (registry import, CI, macOS) this raises and we keep
        # the pipeline object constructible only where it can actually run.
        self._runtime = runtime.build_runtime(
            model_dir=model_dir,
            model_config=self.model_config,
            config=config,
            device=self.device,
            dtype=dtype,
        )

        # Minimal state for cache management across chunks (mirrors longlive2).
        self.state = PipelineState()
        self.state.set("current_start_frame", 0)
        self.state.set("manage_cache", getattr(config, "manage_cache", True))
        self.first_call = True

    def prepare(self, **kwargs) -> Requirements | None:
        """Return input requirements based on current mode (text-only for now)."""
        return prepare_for_mode(self.__class__, self.config, kwargs)

    def __call__(self, **kwargs) -> dict:
        return self._generate(**kwargs)

    def _generate(self, **kwargs) -> dict:
        # Resolve the prompt for this chunk. Scope passes prompts via "prompts"
        # (list) or "prompt"; fall back to empty string.
        # The UI/frame processor sends prompts as a list of
        # {"text": str, "weight": float} dicts; also accept a plain string, a
        # list of strings, or a single dict.
        raw = kwargs.get("prompts") or kwargs.get("prompt")
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else ""
        if isinstance(raw, dict):
            prompt = raw.get("text", "")
        else:
            prompt = raw or ""

        mode = resolve_input_mode(kwargs)  # noqa: F841 - text-only today; reserved
        init_cache = bool(kwargs.get("init_cache", self.first_call))
        self.first_call = False

        # One autoregressive AV pass. Returns the scope AV-dict shape:
        #   { video, video_timestamps, audio, audio_sample_rate,
        #     audio_timestamps, frame_rate }
        # matching daydreamlive/scope-ltx-2's LTX2Pipeline output contract so the
        # (Phase 4) audio-capable WebRTC track can consume it directly.
        return self._runtime.generate_chunk(
            prompt=prompt,
            seed=int(kwargs.get("base_seed", getattr(self.config, "base_seed", 12345))),
            height=int(kwargs.get("height", self.config.height)),
            width=int(kwargs.get("width", self.config.width)),
            num_frames=int(kwargs.get("num_frames", self.config.num_frames)),
            init_cache=init_cache,
        )
