import logging
import os
import time
from typing import TYPE_CHECKING

import torch
import torchvision.transforms.functional as TF
from diffusers.modular_pipelines import PipelineState
from PIL import Image

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
from ..utils import Quantization, load_model_config, validate_resolution
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.vae import create_vae
from .modular_blocks import CausalForcingBlocks
from .schema import CausalForcingConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]


class CausalForcingPipeline(Pipeline):
    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return CausalForcingConfig

    def __init__(
        self,
        config,
        quantization: Quantization | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from ..longlive.modules.causal_model import CausalWanModel

        validate_resolution(
            height=config.height,
            width=config.width,
            scale_factor=16,
        )

        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)

        model_config = load_model_config(config, __file__)
        base_model_name = getattr(model_config, "base_model_name", "Wan2.1-T2V-1.3B")
        base_model_kwargs = getattr(model_config, "base_model_kwargs", {})
        generator_model_name = getattr(
            model_config, "generator_model_name", "generator_ema"
        )

        # Determine if we're in I2V mode based on reference_image config
        reference_image = getattr(config, "reference_image", None)
        is_i2v = reference_image is not None

        # Set model_type based on I2V mode
        if is_i2v:
            base_model_kwargs["model_type"] = "i2v"

        # Load generator
        start = time.time()
        generator = WanDiffusionWrapper(
            CausalWanModel,
            model_name=base_model_name,
            model_dir=model_dir,
            generator_path=generator_path,
            generator_model_name=generator_model_name,
            **base_model_kwargs,
        )

        if quantization == Quantization.FP8_E4M3FN:
            generator = generator.to(dtype=dtype)
            start = time.time()
            from torchao.quantization.quant_api import (
                Float8DynamicActivationFloat8WeightConfig,
                PerTensor,
                quantize_,
            )

            quantize_(
                generator,
                Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                device=device,
            )
            print(f"Quantized diffusion model to fp8 in {time.time() - start:.3f}s")
        else:
            generator = generator.to(device=device, dtype=dtype)
        print(f"Loaded generator in {time.time() - start:.3f}s")

        # Load text encoder
        start = time.time()
        text_encoder = WanTextEncoderWrapper(
            model_name=base_model_name,
            model_dir=model_dir,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        print(f"Loaded text encoder in {time.time() - start:.3f}s")
        text_encoder = text_encoder.to(device=device)

        # Load VAE
        vae_type = getattr(config, "vae_type", "wan")
        start = time.time()
        vae = create_vae(
            model_dir=model_dir, model_name=base_model_name, vae_type=vae_type
        )
        print(f"Loaded VAE (type={vae_type}) in {time.time() - start:.3f}s")
        vae = vae.to(device=device, dtype=dtype)

        # Load CLIP visual encoder for I2V mode
        clip_encoder = None
        if is_i2v:
            from .modules.clip_encoder import WanCLIPVisualEncoder

            clip_path = self._resolve_clip_path(model_dir)
            if clip_path is not None:
                start = time.time()
                clip_encoder = WanCLIPVisualEncoder(
                    checkpoint_path=clip_path,
                    dtype=dtype,
                    device=device,
                )
                clip_encoder = clip_encoder.to(device=device)
                print(f"Loaded CLIP visual encoder in {time.time() - start:.3f}s")
            else:
                logger.warning(
                    "CLIP checkpoint not found. I2V mode will not have CLIP conditioning. "
                    "Download Wan2.1-I2V model for full I2V support."
                )

        # Create components
        components_config = {}
        components_config.update(model_config)
        components_config["device"] = device
        components_config["dtype"] = dtype

        components = ComponentsManager(components_config)
        components.add("generator", generator)
        components.add("scheduler", generator.get_scheduler())
        components.add("vae", vae)
        components.add("text_encoder", text_encoder)
        if clip_encoder is not None:
            components.add("clip_encoder", clip_encoder)

        embedding_blender = EmbeddingBlender(device=device, dtype=dtype)
        components.add("embedding_blender", embedding_blender)

        self.blocks = CausalForcingBlocks()
        self.components = components
        self.state = PipelineState()

        self.state.set("current_start_frame", 0)
        self.state.set("manage_cache", True)
        self.state.set("kv_cache_attention_bias", 1.0)
        self.state.set("height", config.height)
        self.state.set("width", config.width)
        self.state.set("base_seed", getattr(config, "base_seed", 42))

        # Store reference image config for loading in _generate
        self._reference_image_path = reference_image
        self._reference_image_tensor = None
        if reference_image is not None:
            self._reference_image_tensor = self._load_reference_image(
                reference_image, device, dtype
            )

        self.first_call = True
        self.last_mode = None

    @staticmethod
    def _resolve_clip_path(model_dir: str | None) -> str | None:
        """Resolve path to Wan2.1 CLIP checkpoint."""
        model_dir = model_dir if model_dir is not None else "wan_models"
        # Check I2V model directories for CLIP weights
        for model_name in ["Wan2.1-I2V-14B-480P", "Wan2.1-I2V-1.3B-480P"]:
            clip_path = os.path.join(
                model_dir,
                model_name,
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            )
            if os.path.exists(clip_path):
                return clip_path
        return None

    @staticmethod
    def _load_reference_image(
        image_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Load reference image and convert to tensor in [-1, 1] range."""
        img = Image.open(image_path).convert("RGB")
        # Convert to tensor [C, H, W] in [0, 1] then normalize to [-1, 1]
        tensor = TF.to_tensor(img).sub_(0.5).div_(0.5)
        # Add batch dim: [1, C, H, W]
        return tensor.unsqueeze(0).to(device=device, dtype=dtype)

    def prepare(self, **kwargs) -> Requirements | None:
        return prepare_for_mode(self.__class__, self.components.config, kwargs)

    def __call__(self, **kwargs) -> dict:
        self.first_call, self.last_mode = handle_mode_transition(
            self.state, self.components.vae, self.first_call, self.last_mode, kwargs
        )
        return self._generate(**kwargs)

    def _generate(self, **kwargs) -> dict:
        for k, v in kwargs.items():
            self.state.set(k, v)

        # Clear stale state
        if "transition" not in kwargs:
            self.state.set("transition", None)
        if "video" not in kwargs:
            self.state.set("video", None)

        # Set reference image tensor in state for I2V blocks
        if self._reference_image_tensor is not None:
            self.state.set("reference_image_tensor", self._reference_image_tensor)
        else:
            self.state.set("reference_image_tensor", None)

        if self.state.get("denoising_step_list") is None:
            self.state.set("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)

        # Apply mode-specific defaults
        mode = resolve_input_mode(kwargs)
        apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

        _, self.state = self.blocks(self.components, self.state)
        return {"video": postprocess_chunk(self.state.values["output_video"])}
