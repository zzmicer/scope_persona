from diffusers.modular_pipelines import AutoPipelineBlocks, SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict

from ..wan2_1.blocks import (
    AutoPreprocessVideoBlock,
    CleanKVCacheBlock,
    DecodeBlock,
    DenoiseBlock,
    EmbeddingBlendingBlock,
    PrepareLatentsBlock,
    PrepareNextBlock,
    SetTimestepsBlock,
    SetTransformerBlocksLocalAttnSizeBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)
from .blocks import CLIPImageConditioningBlock, PrepareI2VLatentsBlock


class AutoPrepareI2VLatentsBlock(AutoPipelineBlocks):
    """Auto-routing block for I2V vs T2V latent preparation.

    Routes to PrepareI2VLatentsBlock when reference_image_tensor is provided,
    otherwise falls back to standard PrepareLatentsBlock for T2V mode.
    """

    block_classes = [
        PrepareI2VLatentsBlock,
        PrepareLatentsBlock,
    ]

    block_names = [
        "prepare_i2v_latents",
        "prepare_latents",
    ]

    block_trigger_inputs = [
        "reference_image_tensor",
        None,
    ]

    @property
    def description(self):
        return (
            "AutoPrepareI2VLatentsBlock: Routes latent preparation:\n"
            " - Routes to PrepareI2VLatentsBlock when reference image is provided\n"
            " - Routes to PrepareLatentsBlock for standard T2V generation\n"
        )


# Block sequence for Causal-Forcing pipeline
# Supports both T2V (text-only) and I2V (reference image) modes.
# I2V blocks (clip_conditioning, auto_prepare_i2v_latents) are no-ops when no
# reference image is provided, falling through to standard T2V behavior.
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("embedding_blending", EmbeddingBlendingBlock),
        ("clip_conditioning", CLIPImageConditioningBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("setup_caches", SetupCachesBlock),
        (
            "set_transformer_blocks_local_attn_size",
            SetTransformerBlocksLocalAttnSizeBlock,
        ),
        ("auto_preprocess_video", AutoPreprocessVideoBlock),
        ("auto_prepare_i2v_latents", AutoPrepareI2VLatentsBlock),
        ("denoise", DenoiseBlock),
        ("clean_kv_cache", CleanKVCacheBlock),
        ("decode", DecodeBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class CausalForcingBlocks(SequentialPipelineBlocks):
    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())
