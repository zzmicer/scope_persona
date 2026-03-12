"""Helios Distilled pipeline configuration schema."""

from typing import ClassVar

from pydantic import Field

from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


class HeliosConfig(BasePipelineConfig):
    """Configuration for Helios Distilled autoregressive video generation pipeline.

    Helios is a 14B-parameter real-time long video generation model based on Wan2.1.
    The Distilled variant uses pyramid multi-scale sampling with only 2 denoising steps
    per stage, achieving ~19.5 FPS on a single H100 GPU.

    Key features:
    - Autoregressive chunk-based generation (33 pixel frames per chunk)
    - Multi-term memory patchification (long/mid/short history)
    - Pyramid multi-scale denoising for efficiency
    - No classifier-free guidance needed (guidance_scale=1.0)
    """

    pipeline_id: ClassVar[str] = "helios"
    pipeline_name: ClassVar[str] = "Helios Distilled"
    pipeline_description: ClassVar[str] = (
        "Real-time autoregressive video generation with Helios Distilled. "
        "A 14B model achieving ~19.5 FPS on H100 via pyramid multi-scale "
        "sampling with multi-term memory patchification."
    )
    pipeline_version: ClassVar[str] = "0.1.0"
    docs_url: ClassVar[str | None] = "https://github.com/PKU-YuanGroup/Helios"
    estimated_vram_gb: ClassVar[float | None] = 40.0
    requires_models: ClassVar[bool] = True
    artifacts: ClassVar[list] = [
        HuggingfaceRepoArtifact(
            repo_id="BestWishYsh/Helios-Distilled",
            files=[
                "model_index.json",
                "scheduler",
                "text_encoder",
                "tokenizer",
                "transformer",
                "vae",
            ],
        ),
    ]
    supports_lora: ClassVar[bool] = False
    supports_vace: ClassVar[bool] = False

    supports_cache_management: ClassVar[bool] = False
    supports_kv_cache_bias: ClassVar[bool] = False
    supports_quantization: ClassVar[bool] = False
    min_dimension: ClassVar[int] = 16
    modified: ClassVar[bool] = False

    modes: ClassVar[dict[str, ModeDefaults]] = {"text": ModeDefaults(default=True)}

    supports_prompts: ClassVar[bool] = True

    # Resolution — Helios works best at 384x640
    height: int = Field(
        default=384,
        ge=1,
        description="Output height in pixels",
        json_schema_extra=ui_field_config(
            order=4, component="resolution", is_load_param=True
        ),
    )
    width: int = Field(
        default=640,
        ge=1,
        description="Output width in pixels",
        json_schema_extra=ui_field_config(
            order=4, component="resolution", is_load_param=True
        ),
    )

    # Pyramid denoising steps per stage (3 stages)
    pyramid_steps: int = Field(
        default=1,
        ge=1,
        le=20,
        description=(
            "Number of denoising steps per pyramid stage. "
            "Distilled model is designed for 1 step per stage (3 total). "
            "Higher values improve quality at the cost of speed."
        ),
        json_schema_extra=ui_field_config(
            order=6, label="Pyramid Steps", is_load_param=False
        ),
    )

    amplify_first_chunk: bool = Field(
        default=False,
        description=(
            "Amplify the first chunk by doubling denoising steps on chunk=0. "
            "Improves initial quality but doubles first-chunk latency (~3.7s vs ~2.3s), "
            "causing a visible freeze on every prompt change."
        ),
        json_schema_extra=ui_field_config(
            order=7, label="Amplify First Chunk", is_load_param=False
        ),
    )

    offload_text_encoder: bool = Field(
        default=False,
        description=(
            "Offload text encoder to CPU after encoding prompts. "
            "Saves ~8GB VRAM but adds ~6s latency on every prompt change (causes freezes). "
            "Keep False on H100 where VRAM is not a constraint."
        ),
        json_schema_extra=ui_field_config(
            order=8, label="Offload Text Encoder", is_load_param=True
        ),
    )

    compile: bool = Field(
        default=True,
        description=(
            "Enable torch.compile on transformer blocks. "
            "Provides 1.3-1.8x speedup after initial warmup (~30-60s). "
            "Required to reach target FPS on H100."
        ),
        json_schema_extra=ui_field_config(
            order=9, label="Compile Transformer", is_load_param=True
        ),
    )

    attention_backend: str | None = Field(
        default=None,
        description=(
            "Attention backend override. Auto-detects best backend if None "
            "(FA3 for H100+, FA2 for Ampere). Options: '_FLASH_3_HUB', 'FLASH_HUB', or None."
        ),
        json_schema_extra=ui_field_config(
            order=10, label="Attention Backend", is_load_param=True
        ),
    )

    compile_vae: bool = Field(
        default=False,
        description=(
            "Enable torch.compile on VAE decoder. "
            "Moderate speedup for decoding, adds to initial warmup time."
        ),
        json_schema_extra=ui_field_config(
            order=11, label="Compile VAE", is_load_param=True
        ),
    )
