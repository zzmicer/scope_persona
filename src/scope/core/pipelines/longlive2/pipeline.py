import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from diffusers.modular_pipelines import PipelineState
from omegaconf import OmegaConf

from ..blending import EmbeddingBlender
from ..components import ComponentsManager
from ..defaults import (
    apply_mode_defaults_to_state,
    handle_mode_transition,
    prepare_for_mode,
    resolve_input_mode,
)
from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk
from ..utils import Quantization, validate_resolution

# wan2_2 component layer (parallel scaffold). These are the Wan2.2-TI2V-5B
# equivalents of the wan2_1 wrappers used by LongLive 1. The text encoder is
# UMT5 (shared with Wan2.1), so it is imported from wan2_1.
from ..wan2_1.components import WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from ..wan2_2.components import WanDiffusionWrapper
from ..wan2_2.vae import create_vae
from .modular_blocks import LongLive2Blocks
from .schema import PRECISION_ARTIFACTS, LongLive2Config

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]


class LongLive2Pipeline(Pipeline, LoRAEnabledPipeline):
    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return LongLive2Config

    def __init__(
        self,
        config,
        quantization: Quantization | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from ..wan2_2.modules.causal_model import CausalWanModel

        # Validate resolution requirements.
        # Wan2.2 VAE spatial downsample (16) * patch embedding downsample (2) = 32
        validate_resolution(
            height=config.height,
            width=config.width,
            scale_factor=32,
        )

        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)

        precision = getattr(config, "precision", "nvfp4-s2")

        # Load model.yaml directly. NOTE: we cannot use the shared
        # load_model_config() helper here because it probes
        # getattr(config, "model_config"), which collides with Pydantic v2's
        # reserved `model_config = ConfigDict(...)` on BasePipelineConfig. That
        # reserved attribute is truthy, so the helper would return Pydantic's
        # ConfigDict instead of the model.yaml contents, dropping required keys
        # like `max_rope_freq_table_seq_len`. LongLive 1 avoids this only because
        # it passes an OmegaConf config (no `model_config` attribute).
        model_config = OmegaConf.load(Path(__file__).parent / "model.yaml")
        base_model_name = getattr(model_config, "base_model_name", "Wan2.2-TI2V-5B")
        base_model_kwargs = getattr(model_config, "base_model_kwargs", {})
        generator_model_name = getattr(
            model_config, "generator_model_name", "generator"
        )

        # Load generator (Wan2.2-TI2V-5B causal transformer) via the wan2_2 wrapper.
        # Unlike LongLive 1, the performance LoRA is PRE-MERGED into the released
        # inference checkpoints, so there is no separate lora.pt load here. The
        # LoRAEnabledPipeline mixin still allows additional user-configured LoRAs.
        start = time.time()
        generator = WanDiffusionWrapper(
            CausalWanModel,
            model_name=base_model_name,
            model_dir=model_dir,
            generator_path=generator_path,
            generator_model_name=generator_model_name,
            **base_model_kwargs,
        )
        print(f"Loaded generator in {time.time() - start:.3f}s")

        # Initialize any additional, user-configured LoRA adapters via shared manager.
        generator.model = self._init_loras(config, generator.model)

        # ------------------------------------------------------------------
        # NVFP4 quantization (Blackwell-only path).
        #
        # For precision in {"nvfp4-s2", "nvfp4-s4"} the released checkpoints are
        # NVFP4-quantized (model_te.pt = TransformerEngine, model_4o6.pt =
        # FourOverSix). setup_nvfp4_pipeline (ported from upstream NVlabs/LongLive
        # utils/inference_utils.py) installs the NVFP4 linears and loads the
        # quantized weights. nvfp4_available() is False unless CUDA + TE/4o6 are
        # importable, so on non-Blackwell hosts (incl. macOS) we cleanly fall back
        # to a bf16 cast instead of erroring.
        #
        # The active artifact (repo + files) is PRECISION_ARTIFACTS[precision].
        #
        # NOTE: setup_nvfp4_pipeline keeps upstream's config-driven contract
        # (model_quant_* keys, te/4o6 paths). The exact config-attribute mapping
        # is reconciled on the Blackwell pod against the real checkpoints; any
        # mismatch falls back to bf16 here rather than crashing the load.
        # ------------------------------------------------------------------
        _active_generator_artifact = PRECISION_ARTIFACTS.get(precision)  # noqa: F841

        if precision in ("nvfp4-s2", "nvfp4-s4"):
            from ..wan2_2.nvfp4 import nvfp4_available, setup_nvfp4_pipeline

            if nvfp4_available():
                try:
                    start = time.time()
                    generator = setup_nvfp4_pipeline(generator, model_config, device)
                    print(
                        f"Set up NVFP4 ({precision}) generator in "
                        f"{time.time() - start:.3f}s"
                    )
                except Exception as exc:  # pragma: no cover - GPU-only path
                    logger.warning(
                        "LongLive2: NVFP4 setup for '%s' failed (%s); falling "
                        "back to bf16 cast.",
                        precision,
                        exc,
                    )
                    generator = generator.to(device=device, dtype=dtype)
            else:
                logger.warning(
                    "LongLive2: NVFP4 precision '%s' selected but the NVFP4 "
                    "runtime is unavailable on this host (needs Blackwell + "
                    "transformer-engine/fouroversix). Falling back to bf16 cast.",
                    precision,
                )
                generator = generator.to(device=device, dtype=dtype)
        else:
            generator = generator.to(device=device, dtype=dtype)

        start = time.time()
        text_encoder = WanTextEncoderWrapper(
            model_name=base_model_name,
            model_dir=model_dir,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        print(f"Loaded text encoder in {time.time() - start:3f}s")
        # Move text encoder to target device but use dtype of weights
        text_encoder = text_encoder.to(device=device)

        # Load Wan2.2 VAE (48-channel latents) via the wan2_2 create_vae factory.
        start = time.time()
        vae = create_vae(model_dir=model_dir, model_name=base_model_name)
        print(f"Loaded VAE in {time.time() - start:.3f}s")
        vae = vae.to(device=device, dtype=dtype)

        # Create components config
        components_config = {}
        components_config.update(model_config)
        components_config["device"] = device
        components_config["dtype"] = dtype

        components = ComponentsManager(components_config)
        components.add("generator", generator)
        components.add("scheduler", generator.get_scheduler())
        components.add("vae", vae)
        components.add("text_encoder", text_encoder)

        embedding_blender = EmbeddingBlender(
            device=device,
            dtype=dtype,
        )
        components.add("embedding_blender", embedding_blender)

        self.blocks = LongLive2Blocks()
        self.components = components
        self.state = PipelineState()
        # These need to be set right now because InputParam.default on the blocks
        # does not work properly
        self.state.set("current_start_frame", 0)
        self.state.set("manage_cache", True)
        self.state.set("kv_cache_attention_bias", 1.0)

        self.state.set("height", config.height)
        self.state.set("width", config.width)
        self.state.set("base_seed", getattr(config, "base_seed", 42))

        # Resolve effective denoising steps from precision (or explicit override).
        num_steps = config.resolve_steps()
        self.state.set("steps", num_steps)

        # Precompute the DMD denoising-timestep SCHEDULE (the actual timestep
        # values, not just the count). LongLive 2.0's 5B checkpoints are distilled
        # for a fixed schedule; the base 4-step schedule comes from model.yaml
        # (`denoising_steps`). For fewer steps (e.g. nvfp4-s2 = 2) we evenly
        # subsample it. The generic frontend default (e.g. [700, 500], tuned for
        # the 1.3B LongLive 1) does NOT match and yields pure noise, so _generate
        # enforces this schedule whenever the incoming list length is wrong.
        base_schedule = list(
            getattr(model_config, "denoising_steps", DEFAULT_DENOISING_STEP_LIST)
        )
        if num_steps >= len(base_schedule):
            self._denoising_step_list = base_schedule
        else:
            # Evenly sample `num_steps` timesteps from the base schedule, always
            # keeping the highest-noise step first.
            idx = (
                [
                    round(i * (len(base_schedule) - 1) / (num_steps - 1))
                    for i in range(num_steps)
                ]
                if num_steps > 1
                else [0]
            )
            self._denoising_step_list = [base_schedule[i] for i in idx]

        self.first_call = True
        self.last_mode = None  # Track mode for transition detection

    def prepare(self, **kwargs) -> Requirements | None:
        """Return input requirements based on current mode."""
        return prepare_for_mode(self.__class__, self.components.config, kwargs)

    def __call__(self, **kwargs) -> dict:
        self.first_call, self.last_mode = handle_mode_transition(
            self.state, self.components.vae, self.first_call, self.last_mode, kwargs
        )
        return self._generate(**kwargs)

    def _generate(self, **kwargs) -> dict:
        # Handle runtime LoRA scale updates before writing into state.
        lora_scales = kwargs.get("lora_scales")
        if lora_scales is not None:
            self._handle_lora_scale_updates(
                lora_scales=lora_scales, model=self.components.generator.model
            )
            # Trigger cache reset on LoRA scale updates if manage_cache is enabled
            if self.state.get("manage_cache", True):
                kwargs["init_cache"] = True

        for k, v in kwargs.items():
            self.state.set(k, v)

        # Clear transition from state if not provided to prevent stale transitions
        if "transition" not in kwargs:
            self.state.set("transition", None)

        # Clear video from state if not provided to prevent stale video data
        if "video" not in kwargs:
            self.state.set("video", None)

        # Clear image-mode first-frame conditioning if not provided to prevent
        # reuse on non-first chunks.
        if "first_frame_image" not in kwargs:
            self.state.set("first_frame_image", None)

        # Enforce the precision-correct DMD schedule. The frontend default
        # (e.g. [700, 500]) is a 2-step list tuned for the 1.3B LongLive 1 and
        # produces pure noise on the 5B model. Override it whenever the incoming
        # list is missing or has the wrong length for this precision.
        incoming_steps = self.state.get("denoising_step_list")
        if not incoming_steps or len(incoming_steps) != len(self._denoising_step_list):
            self.state.set("denoising_step_list", list(self._denoising_step_list))

        # Apply mode-specific defaults
        mode = resolve_input_mode(kwargs)
        apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

        _, self.state = self.blocks(self.components, self.state)
        return {"video": postprocess_chunk(self.state.values["output_video"])}
