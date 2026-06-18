from typing import Literal

from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, ui_field_config
from ..common_artifacts import UMT5_ENCODER_ARTIFACT

# ---------------------------------------------------------------------------
# Artifacts (defined inline per LongLive 2.0 scaffold constraints).
#
# All HF repos below are referenced from the LongLive 2.0 release
# (Efficient-Large-Model org) and the Wan2.2-TI2V-5B base model. File lists for
# the base model were verified against the HF API:
#   curl -s https://huggingface.co/api/models/Wan-AI/Wan2.2-TI2V-5B
# The Wan2.2 VAE filename is "Wan2.2_VAE.pth" and the tokenizer lives under the
# "google/" directory (umt5-xxl), mirroring the wan2_1 WAN_1_3B_ARTIFACT shape.
# ---------------------------------------------------------------------------

# Base Wan2.2-TI2V-5B model: config + 48-channel Wan2.2 VAE + umt5 tokenizer dir.
WAN_2_2_TI2V_5B_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    files=["config.json", "Wan2.2_VAE.pth", "google"],
)

# NVFP4 generator (2-step / S2) — fastest, Blackwell (RTX 50xx / 5090) only.
# Files: text-encoder-side merged weights ("model_te.pt") + the 4-over-6
# quantized transformer weights ("model_4o6.pt").
LONGLIVE2_NVFP4_S2_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2",
    files=["model_te.pt", "model_4o6.pt"],
)

# NVFP4 generator (4-step / S4) — higher quality, also Blackwell-only.
LONGLIVE2_NVFP4_S4_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4",
    files=["model_te.pt", "model_4o6.pt"],
)

# BF16 generator — portable fallback (no NVFP4 hardware required, higher VRAM).
LONGLIVE2_BF16_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Efficient-Large-Model/LongLive-2.0-5B",
    files=["model_bf16.pt"],
)


# Map precision -> (artifact, default number of denoising steps).
# NVFP4-S2 is 2-step, NVFP4-S4 and BF16 use the 4-step DMD schedule.
PRECISION_ARTIFACTS = {
    "nvfp4-s2": LONGLIVE2_NVFP4_S2_ARTIFACT,
    "nvfp4-s4": LONGLIVE2_NVFP4_S4_ARTIFACT,
    "bf16": LONGLIVE2_BF16_ARTIFACT,
}

PRECISION_STEPS = {
    "nvfp4-s2": 2,
    "nvfp4-s4": 4,
    "bf16": 4,
}


class LongLive2Config(BasePipelineConfig):
    pipeline_id = "longlive2"
    pipeline_name = "LongLive 2.0"
    pipeline_description = (
        "LongLive 2.0 is a streaming autoregressive video diffusion model built on "
        "Wan2.2-TI2V-5B. It supports both text-to-video and image-to-video (first-frame "
        "conditioning) and ships NVFP4 quantized checkpoints for real-time generation on "
        "Blackwell GPUs (RTX 5090). The S2 (2-step) checkpoint targets ~45fps while the "
        "S4 (4-step) checkpoint trades speed for quality."
    )
    docs_url = "https://github.com/daydreamlive/scope/blob/main/src/scope/core/pipelines/longlive2/docs/usage.md"
    # 5B NVFP4 weights are compact; transformer + VAE + text encoder fit in ~12GB.
    estimated_vram_gb = 12.0
    supports_lora = True
    supports_vace = False
    # Auto-downloaded weights (download_models --pipeline longlive2). Kept minimal:
    # base + text encoder + the default NVFP4-S2 generator (the Blackwell/5090
    # target). The S4 and BF16 generators are defined in PRECISION_ARTIFACTS and
    # are fetched on demand only if the user switches `precision`, so we don't pull
    # ~20GB of extra checkpoints onto the pod by default.
    artifacts = [
        WAN_2_2_TI2V_5B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        LONGLIVE2_NVFP4_S2_ARTIFACT,
    ]

    supports_cache_management = True
    supports_quantization = True
    # VAE downsample (16) * patch embedding downsample (2) = 32.
    min_dimension = 32
    modified = True

    # Configuration fields with UI metadata (order, component, modes)
    precision: Literal["nvfp4-s2", "nvfp4-s4", "bf16"] = Field(
        default="nvfp4-s2",
        description=(
            "Generator precision/checkpoint. 'nvfp4-s2' (2-step) is fastest (~45fps, "
            "Blackwell only), 'nvfp4-s4' (4-step) is higher quality (Blackwell only), "
            "'bf16' is the portable fallback."
        ),
        json_schema_extra=ui_field_config(
            order=1, is_load_param=True, label="Precision"
        ),
    )
    height: int = Field(
        default=704,
        ge=1,
        description="Output height in pixels",
        json_schema_extra=ui_field_config(
            order=2, component="resolution", is_load_param=True
        ),
    )
    width: int = Field(
        default=1280,
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
    # Number of denoising steps. None -> derive from `precision` (see PRECISION_STEPS).
    # Explicit override allowed for experimentation.
    steps: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of denoising steps. Leave unset to derive from precision "
            "(nvfp4-s2 -> 2, nvfp4-s4 / bf16 -> 4)."
        ),
        json_schema_extra=ui_field_config(order=5, is_load_param=True, label="Steps"),
    )
    # NVFP4 backend selection. Default FourOverSix (model_4o6.pt) is the validated
    # path. TransformerEngine (model_te.pt) is EXPERIMENTAL/unverified in Scope —
    # it requires the `transformer-engine` wheel on a Blackwell host and has not
    # been render-verified here. Selecting it switches both the construction
    # checkpoint (model_te.pt) and the setup_nvfp4_pipeline quantization path.
    model_quant_use_transformer_engine: bool = Field(
        default=False,
        description=(
            "Use the TransformerEngine NVFP4 backend (model_te.pt) instead of the "
            "default FourOverSix backend (model_4o6.pt). Experimental: requires "
            "transformer-engine on Blackwell and is not render-verified."
        ),
        json_schema_extra=ui_field_config(
            order=6, is_load_param=True, label="Use TransformerEngine (experimental)"
        ),
    )

    modes = {
        # TI2V supports BOTH text-to-video and image-to-video. Image mode uses
        # first-frame conditioning (the supplied image is encoded as frame 0).
        "text": ModeDefaults(default=True),
        "image": ModeDefaults(),
    }

    def resolve_steps(self) -> int:
        """Return effective denoising steps: explicit override or precision default."""
        if self.steps is not None:
            return self.steps
        return PRECISION_STEPS[self.precision]
