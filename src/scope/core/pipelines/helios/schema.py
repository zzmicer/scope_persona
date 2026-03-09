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
        default=2,
        ge=1,
        le=20,
        description=(
            "Number of denoising steps per pyramid stage. "
            "Distilled model works well with 2 steps. "
            "Higher values improve quality at the cost of speed."
        ),
        json_schema_extra=ui_field_config(
            order=6, label="Pyramid Steps", is_load_param=False
        ),
    )

    amplify_first_chunk: bool = Field(
        default=True,
        description=(
            "Amplify the first chunk for better initial quality. "
            "Recommended for the distilled model."
        ),
        json_schema_extra=ui_field_config(
            order=7, label="Amplify First Chunk", is_load_param=False
        ),
    )

    offload_text_encoder: bool = Field(
        default=True,
        description=(
            "Offload text encoder to CPU after encoding prompts. "
            "Saves significant VRAM but adds latency when prompts change."
        ),
        json_schema_extra=ui_field_config(
            order=8, label="Offload Text Encoder", is_load_param=True
        ),
    )
