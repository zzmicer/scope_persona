from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

# The wan2_1 modular blocks operate on the shared pipeline State and are largely
# model-agnostic (they call into components.generator / vae / scheduler), so the
# LongLive 2.0 (Wan2.2-TI2V-5B) pipeline reuses them. Blocks that genuinely
# differ for the 5B / NVFP4 / multi-shot path are marked with TODOs below.
from ..wan2_1.blocks import (
    AutoPrepareLatentsBlock,
    AutoPreprocessVideoBlock,
    CleanKVCacheBlock,
    DecodeBlock,
    DenoiseBlock,
    EmbeddingBlendingBlock,
    PrepareNextBlock,
    SetTimestepsBlock,
    SetTransformerBlocksLocalAttnSizeBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)

# NOTE: LongLive 2.0 reuses the LongLive 1 recache blocks for now. The 5B model's
# multi-shot prompt switching (multi_shot_sink / multi_shot_rope_offset) may need
# a dedicated recache/rope-offset block.
# TODO(longlive2-phase3): if 5B multi-shot diverges from 1.3B recaching, port
# upstream NVlabs/LongLive wan_5b multi-shot KV-cache handling into local blocks
# (mirror longlive/blocks/{prepare_recache_frames,recache_frames}.py) instead of
# importing the 1.3B versions below.
from ..longlive.blocks import (
    PrepareRecacheFramesBlock,
    RecacheFramesBlock,
)

logger = diffusers_logging.get_logger(__name__)

# Main pipeline blocks with multi-mode support (text-to-video and image-to-video).
# AutoPreprocessVideoBlock / AutoPrepareLatentsBlock route image-mode first-frame
# conditioning vs pure text-to-video. VACE blocks are intentionally absent
# (LongLive 2.0 does not ship a VACE module).
#
# TODO(longlive2-phase3): verify DenoiseBlock honors the 2-step (NVFP4-S2) vs
# 4-step (NVFP4-S4 / BF16) schedule selected from `precision`; the schedule is
# placed on state as `steps` by the pipeline. The 5B DMD timesteps must come
# from upstream wan_5b/configs/wan_ti2v_5B.py, not the 1.3B defaults.
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("embedding_blending", EmbeddingBlendingBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("setup_caches", SetupCachesBlock),
        (
            "set_transformer_blocks_local_attn_size",
            SetTransformerBlocksLocalAttnSizeBlock,
        ),
        ("auto_preprocess_video", AutoPreprocessVideoBlock),
        ("auto_prepare_latents", AutoPrepareLatentsBlock),
        ("recache_frames", RecacheFramesBlock),
        ("denoise", DenoiseBlock),
        ("clean_kv_cache", CleanKVCacheBlock),
        ("decode", DecodeBlock),
        ("prepare_recache_frames", PrepareRecacheFramesBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class LongLive2Blocks(SequentialPipelineBlocks):
    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())
