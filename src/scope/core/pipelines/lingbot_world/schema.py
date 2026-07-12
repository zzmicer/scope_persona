"""Config + artifact definitions for the LingBot-World-V2 interactive world pipeline.

LingBot-World-V2 (https://github.com/Robbyant/lingbot-world-v2) is a Wan2.2-based
14B *causal* world model: image -> interactive world. Camera motion is conditioned
on per-latent-frame framewise pose deltas (Plücker embeddings); scene content and
events are driven by the text prompt; generation is chunk-by-chunk over a rolling
KV cache (local attention window + sink), giving an unbounded horizon.

This module follows the ``omniforcing/schema.py`` shape: inline artifact
definition + a ``BasePipelineConfig`` subclass with UI field metadata. The
upstream model code (``wan`` package) is NOT vendored (CC BY-NC-SA 4.0); it is
imported at runtime from a checkout pointed to by ``lingbot_repo`` /
``LINGBOT_WORLD_REPO``.
"""

from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, CtrlInput, ModeDefaults, ui_field_config

# Verified file list (pod download, 2026-07-12). Files are listed explicitly —
# the download-status check resolves each entry to a literal Path (no globs).
LINGBOT_WORLD_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="robbyant/lingbot-world-v2-14b-causal-fast",
    files=[
        "config.json",
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl/special_tokens_map.json",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json",
        "transformers/diffusion_pytorch_model.safetensors.index.json",
        *[f"transformers/model-{i:05d}-of-00008.safetensors" for i in range(1, 9)],
    ],
)


class LingbotWorldConfig(BasePipelineConfig):
    pipeline_id = "lingbot-world"
    pipeline_name = "LingBot World (interactive)"
    pipeline_description = (
        "LingBot-World-V2 (14B causal-fast, Wan2.2-based) interactive world model: "
        "turn a single start image into an explorable world. Drive the camera with "
        "WASD + arrow keys (or Q/E to orbit); the text prompt controls the scene "
        "and character events (e.g. 'she waves at the camera'). ~70GB VRAM in "
        "bf16 — H100/H200 class. Non-commercial license (CC BY-NC-SA 4.0)."
    )
    docs_url = (
        "https://github.com/daydreamlive/scope/blob/main/src/scope/core/"
        "pipelines/lingbot_world/README.md"
    )
    estimated_vram_gb = 70.0
    supports_lora = False
    supports_vace = False
    supports_cache_management = True
    supports_quantization = False
    # Wan2.1 VAE stride 8 x patch 2 -> height/width multiples of 16.
    min_dimension = 16
    # Integration of external (CC BY-NC-SA) model code, loaded from lingbot_repo.
    modified = True
    artifacts = [LINGBOT_WORLD_ARTIFACT]

    # Text mode: no input video stream is consumed; the world starts from
    # first_frame_image (set it in the UI before or during streaming).
    modes = {"text": ModeDefaults(default=True)}

    # Presence of this field enables frontend keyboard/mouse capture.
    ctrl_input: CtrlInput | None = None

    height: int = Field(
        default=480,
        ge=16,
        description="Output height in pixels (multiple of 16)",
        json_schema_extra=ui_field_config(
            order=1, component="resolution", is_load_param=True
        ),
    )
    width: int = Field(
        default=832,
        ge=16,
        description="Output width in pixels (multiple of 16)",
        json_schema_extra=ui_field_config(
            order=2, component="resolution", is_load_param=True
        ),
    )
    lingbot_repo: str = Field(
        default="/workspace/lingbot-world-v2",
        description=(
            "Path to a lingbot-world-v2 checkout (provides the `wan` package). "
            "Overridable via LINGBOT_WORLD_REPO."
        ),
        json_schema_extra=ui_field_config(order=10, is_load_param=True),
    )
    ckpt_dir: str = Field(
        default="",
        description=(
            "Checkpoint directory (lingbot-world-v2-14b-causal-fast). Empty = "
            "<models_dir>/lingbot-world-v2-14b-causal-fast."
        ),
        json_schema_extra=ui_field_config(order=11, is_load_param=True),
    )
    start_image: str = Field(
        default="",
        description=(
            "Path to the start image that seeds the world. Can be replaced "
            "mid-stream via the first-frame image picker."
        ),
        json_schema_extra=ui_field_config(order=3, component="image"),
    )
    chunk_size: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Latent frames per generation step (4 latent = 16 video frames)",
        json_schema_extra=ui_field_config(order=12, is_load_param=True),
    )
    local_attn_size: int = Field(
        default=18,
        description="KV-cache attention window in latent frames",
        json_schema_extra=ui_field_config(order=13, is_load_param=True),
    )
    sink_size: int = Field(
        default=6,
        description="Attention-sink size in latent frames",
        json_schema_extra=ui_field_config(order=14, is_load_param=True),
    )
    max_frames: int = Field(
        default=641,
        description=(
            "Image-conditioning horizon in video frames (~40s). When exhausted, "
            "the session transparently re-seeds from the last generated frame."
        ),
        json_schema_extra=ui_field_config(order=15, is_load_param=True),
    )
    move_speed: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Translation per latent frame for WASD movement (0-1)",
        json_schema_extra=ui_field_config(order=20),
    )
    turn_speed: float = Field(
        default=4.5,
        ge=0.0,
        le=20.0,
        description="Rotation in degrees per latent frame for turning/looking",
        json_schema_extra=ui_field_config(order=21),
    )
    base_seed: int = Field(
        default=42,
        description="Random seed",
        json_schema_extra=ui_field_config(order=30),
    )
