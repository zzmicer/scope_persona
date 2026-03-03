from pydantic import Field

from ..base_schema import BasePipelineConfig, ModeDefaults, ui_field_config
from ..common_artifacts import (
    CAUSAL_FORCING_ARTIFACT,
    LIGHTTAE_ARTIFACT,
    LIGHTVAE_ARTIFACT,
    TAE_ARTIFACT,
    UMT5_ENCODER_ARTIFACT,
    WAN_1_3B_ARTIFACT,
    WAN_I2V_1_3B_CLIP_ARTIFACT,
)
from ..enums import Quantization
from ..utils import VaeType


class CausalForcingConfig(BasePipelineConfig):
    pipeline_id = "causal_forcing"
    pipeline_name = "Causal Forcing"
    pipeline_description = (
        "Real-time autoregressive video generation with image-to-video support, "
        "based on Causal-Forcing distillation of Wan2.1. Supports few-step inference "
        "for fast streaming and reference image conditioning for character-consistent generation."
    )
    estimated_vram_gb = 20.0
    supports_lora = False
    supports_vace = False
    supports_cache_management = True
    supports_quantization = True
    min_dimension = 16
    modified = True

    artifacts = [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        LIGHTVAE_ARTIFACT,
        TAE_ARTIFACT,
        LIGHTTAE_ARTIFACT,
        CAUSAL_FORCING_ARTIFACT,
        WAN_I2V_1_3B_CLIP_ARTIFACT,
    ]

    # Configuration fields
    vae_type: VaeType = Field(
        default=VaeType.WAN,
        description="VAE type to use. 'wan' is the full VAE, 'lightvae' is 75% pruned (faster but lower quality).",
        json_schema_extra=ui_field_config(order=1, is_load_param=True, label="VAE"),
    )
    height: int = Field(
        default=480,
        ge=1,
        description="Output height in pixels",
        json_schema_extra=ui_field_config(
            order=2, component="resolution", is_load_param=True
        ),
    )
    width: int = Field(
        default=832,
        ge=1,
        description="Output width in pixels",
        json_schema_extra=ui_field_config(
            order=2, component="resolution", is_load_param=True
        ),
    )
    base_seed: int = Field(
        default=42,
        ge=0,
        description="Base random seed for reproducible generation",
        json_schema_extra=ui_field_config(order=3, is_load_param=True, label="Seed"),
    )
    manage_cache: bool = Field(
        default=True,
        description="Enable automatic cache management for performance optimization",
        json_schema_extra=ui_field_config(
            order=4, component="cache", is_load_param=True
        ),
    )
    denoising_steps: list[int] = Field(
        default=[1000, 750, 500, 250],
        description="Denoising step schedule for progressive generation",
        json_schema_extra=ui_field_config(
            order=5, component="denoising_steps", is_load_param=True
        ),
    )
    reference_image: str | None = Field(
        default=None,
        description="Path to reference image for image-to-video generation",
        json_schema_extra=ui_field_config(
            order=6, component="image", category="input", label="Reference Image"
        ),
    )
    noise_scale: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Amount of noise to add during video generation (video mode only)",
        json_schema_extra=ui_field_config(
            order=7, component="noise", modes=["video"], is_load_param=True
        ),
    )
    noise_controller: bool = Field(
        default=True,
        description="Enable dynamic noise control during generation (video mode only)",
        json_schema_extra=ui_field_config(
            order=7, component="noise", modes=["video"], is_load_param=True
        ),
    )
    quantization: Quantization | None = Field(
        default=None,
        description="Quantization method for the diffusion model.",
        json_schema_extra=ui_field_config(
            order=8, component="quantization", is_load_param=True
        ),
    )

    modes = {
        "text": ModeDefaults(default=True),
        "video": ModeDefaults(
            height=512,
            width=512,
            noise_scale=0.7,
            noise_controller=True,
            denoising_steps=[1000, 750],
        ),
    }
