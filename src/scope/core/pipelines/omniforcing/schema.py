"""Config + artifact definitions for the OmniForcing (LTX-2 causal AV) pipeline.

OmniForcing (https://github.com/OmniForcing/OmniForcing) distills the bidirectional
audio-visual **LTX-2** diffusion model into a *streaming autoregressive* (causal-forcing)
generator: block-causal attention + KV-cache + a short distilled denoising schedule,
producing synchronized audio + video. Conceptually this is the same paradigm as the
``longlive``/``longlive2`` pipelines, but on the LTX-2 family and with an audio track.

This module follows the ``longlive2/schema.py`` shape (inline ``HuggingfaceRepoArtifact``
definitions + a ``BasePipelineConfig`` subclass with UI field metadata).
"""

from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, ui_field_config

# ---------------------------------------------------------------------------
# Artifacts.
#
# Verified against the HF API (2026-06-18):
#   curl -s https://huggingface.co/api/models/Exploration/omniforcing-ltx2-5s-causal
#   curl -s https://huggingface.co/api/models/Lightricks/LTX-2
#
# OmniForcing ships ONLY the distilled causal generator (5 safetensors shards +
# index). The LTX-2 base transformer, video VAE, audio VAE, vocoder, text
# projection (connectors) AND the Gemma-3 text encoder all live in the
# Lightricks/LTX-2 diffusers repo, so a single base artifact covers them.
# NOTE: Lightricks/LTX-2 is PUBLIC (verified 2026-06-19, HTTP 200, gated=false) —
# no HF_TOKEN is required.
# ---------------------------------------------------------------------------

# OmniForcing distilled causal generator (5s, LTX-2). Sharded safetensors.
OMNIFORCING_GENERATOR_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Exploration/omniforcing-ltx2-5s-causal",
    files=[
        "omniforcing_ltx2_5s_causal.safetensors.index.json",
        "omniforcing_ltx2_5s_causal-00001-of-00005.safetensors",
        "omniforcing_ltx2_5s_causal-00002-of-00005.safetensors",
        "omniforcing_ltx2_5s_causal-00003-of-00005.safetensors",
        "omniforcing_ltx2_5s_causal-00004-of-00005.safetensors",
        "omniforcing_ltx2_5s_causal-00005-of-00005.safetensors",
    ],
)

# LTX-2 base. Minimal inference subset CONFIRMED on an H100 pod (2026-06-19):
#   - ltx-2-19b-dev.safetensors : consolidated base — transformer + video VAE +
#     audio VAE + vocoder + connectors (create_vae_wrappers reads the vocoder from
#     THIS file, so the per-component vae/ audio_vae/ vocoder/ connectors/ dirs are
#     NOT needed).
#   - text_encoder/ (model-*.safetensors only) + tokenizer/ : the Gemma-3 text
#     encoder. The loader's gemma_root_path is the LTX-2 *root* and does a
#     recursive rglob for tokenizer.model / preprocessor_config.json / model*.
#     safetensors, so both dirs must sit under the same root. We fetch only the
#     `model-*` shard set (the duplicate `diffusion_pytorch_model-*` set in
#     text_encoder/ is ~24GB and unused).
LTX2_BASE_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Lightricks/LTX-2",
    files=[
        "ltx-2-19b-dev.safetensors",
        "model_index.json",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-*.safetensors",
        "tokenizer",
    ],
)


class OmniForcingConfig(BasePipelineConfig):
    pipeline_id = "omniforcing"
    pipeline_name = "OmniForcing (LTX-2 AV)"
    pipeline_description = (
        "OmniForcing is a streaming autoregressive audio-video diffusion model "
        "distilled from the bidirectional LTX-2 (19B: 14B video + 5B audio) "
        "teacher. It generates synchronized audio + video from a text prompt via "
        "block-causal attention + KV-cache, using a Gemma-3-12B text encoder. "
        "Large model (~60GB bf16): targets datacenter GPUs (H100/H200 80GB), not "
        "consumer Blackwell."
    )
    docs_url = (
        "https://github.com/daydreamlive/scope/blob/main/src/scope/core/"
        "pipelines/omniforcing/docs/usage.md"
    )
    # 19B LTX-2 base (~38GB bf16) + 12B Gemma (~24GB) + VAEs/vocoder. H100/H200 class.
    estimated_vram_gb = 64.0
    supports_lora = False
    supports_vace = False
    supports_cache_management = True
    supports_quantization = False
    # LTX-2 latent patch grid: VAE spatial downsample (32) — height/width multiples of 32.
    min_dimension = 32
    # This pipeline is an integration of external (LTX-2 Community License) model code.
    modified = True
    # Auto-downloaded weights (download_models --pipeline omniforcing): the
    # distilled causal generator + the (public) LTX-2 base (transformer + VAEs +
    # vocoder + Gemma text encoder). No HF_TOKEN required.
    artifacts = [
        OMNIFORCING_GENERATOR_ARTIFACT,
        LTX2_BASE_ARTIFACT,
    ]

    # --- configuration fields (UI metadata mirrors longlive2) ---------------
    height: int = Field(
        default=512,
        ge=1,
        description="Output height in pixels (multiple of 32)",
        json_schema_extra=ui_field_config(
            order=1, component="resolution", is_load_param=True
        ),
    )
    width: int = Field(
        default=768,
        ge=1,
        description="Output width in pixels (multiple of 32)",
        json_schema_extra=ui_field_config(
            order=1, component="resolution", is_load_param=True
        ),
    )
    num_frames: int = Field(
        default=121,
        ge=9,
        le=257,
        description=(
            "Number of video frames to generate (N*8+1 format). Default 121 = "
            "~5s at 24fps, matching the released 5s causal checkpoint."
        ),
        json_schema_extra=ui_field_config(order=2, is_load_param=True, label="Frames"),
    )
    fps: int = Field(
        default=24,
        ge=1,
        description="Output frame rate (the 5s causal checkpoint is trained at 24fps)",
        json_schema_extra=ui_field_config(order=3, is_load_param=True, label="FPS"),
    )
    base_seed: int = Field(
        default=12345,
        ge=0,
        description="Base random seed for reproducible generation",
        json_schema_extra=ui_field_config(order=4, is_load_param=True, label="Seed"),
    )
    # OmniForcing causal block sizes (from omniforcing_causal_inference.py defaults).
    num_frame_per_block: int = Field(
        default=3,
        ge=1,
        description="Causal block size (frames denoised per autoregressive block)",
        json_schema_extra=ui_field_config(order=5, is_load_param=True),
    )
    num_frame_per_block_first: int = Field(
        default=4,
        ge=1,
        description="Causal block size for the FIRST block (priming chunk)",
        json_schema_extra=ui_field_config(order=5, is_load_param=True),
    )
    audio_sample_rate: int = Field(
        default=24000,
        ge=1,
        description="Audio sample rate (Hz) of the generated waveform",
        json_schema_extra=ui_field_config(order=6, is_load_param=True),
    )

    # OmniForcing's distilled per-block denoising schedule (timesteps). The
    # upstream default is [1000, 909, 725, 421, 0] (the trailing 0 is the clean
    # boundary). Exposed as a knob; verify empirically on the pod.
    denoising_steps: list[int] | None = Field(
        default=[1000, 909, 725, 421, 0],
        description=(
            "Distilled per-block denoising timestep schedule. Upstream default "
            "[1000, 909, 725, 421, 0] (trailing 0 = clean boundary)."
        ),
        json_schema_extra=ui_field_config(
            order=7, component="denoising_steps", is_load_param=True
        ),
    )

    # Text-only for now (the 5s causal checkpoint is text->AV). Image/audio-driven
    # conditioning exists upstream and can be added as follow-up modes.
    modes = {
        "text": ModeDefaults(default=True),
    }
