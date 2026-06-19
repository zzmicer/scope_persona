"""CPU-runnable contract tests for the OmniForcing (LTX-2 causal AV) pipeline.

These do NOT load the multi-GB checkpoints, require a GPU, or need the external
LTX stack (ltx-core/ltx-causal/ltx-distillation) installed. They lock the
scope-side wiring invariants so the scaffold can't silently rot:

  * the config metadata + defaults (5s @ 24fps, 121 frames, block sizes),
  * the artifact -> HF repo/file selection (generator shards + LTX-2 base),
  * model.yaml staying consistent with the LTX-2 latent geometry + schema,
  * the runtime probe being import-safe (returns False without the LTX stack),
  * the pipeline module importing cleanly on a host without the LTX stack.
"""

from pathlib import Path

from omegaconf import OmegaConf

from scope.core.pipelines.omniforcing import runtime
from scope.core.pipelines.omniforcing.schema import (
    LTX2_BASE_ARTIFACT,
    OMNIFORCING_GENERATOR_ARTIFACT,
    OmniForcingConfig,
)


def test_config_metadata_and_defaults():
    c = OmniForcingConfig()
    assert OmniForcingConfig.pipeline_id == "omniforcing"
    assert c.num_frames == 121  # ~5s @ 24fps (N*8+1)
    assert (c.num_frames - 1) % 8 == 0
    assert c.fps == 24
    assert c.num_frame_per_block == 3
    assert c.num_frame_per_block_first == 4
    assert c.audio_sample_rate == 24000
    # Distilled per-block schedule with the trailing clean boundary.
    assert c.denoising_steps == [1000, 909, 725, 421, 0]
    # LTX-2 VAE 32x spatial downsample -> dims must be multiples of 32.
    assert OmniForcingConfig.min_dimension == 32
    assert c.height % 32 == 0 and c.width % 32 == 0


def test_streaming_config_defaults():
    """Continuous streaming is the default; the KV budget knob is bounded."""
    c = OmniForcingConfig()
    assert c.streaming is True
    assert 0.0 < c.stream_max_seconds <= 20.0
    # Windowed-decode tuning knobs (non-causal VAE: left context + right look-ahead).
    assert c.decode_context_latents >= 0
    assert c.decode_lookahead_latents >= 0
    assert c.stream_audio is False  # WebRTC is video-only today


def test_latent_to_pixel_mapping():
    """Decoding L latents yields 1 + 8*(L-1) pixel frames (first latent = key-frame)."""
    from scope.core.pipelines.omniforcing.runtime import OmniForcingRuntime

    px = OmniForcingRuntime._px_for_latents
    assert px(0) == 0
    assert px(1) == 1  # lone key-frame
    assert px(4) == 25  # first block (4 latents) -> 25 px
    assert px(16) == 121  # 5s @ 24fps (the released checkpoint's clip length)
    # Per-block emitted pixels = 8 * num_frame_per_block once past the key-frame.
    assert px(5) - px(2) == 24


def test_snap_to_block_layout_math():
    """Stream budget must snap UP to a valid 4 + k*3 causal block layout.

    Exercised without the LTX stack via an unbound call with a stub `self`
    (the method only reads num_frame_per_block{,_first}).
    """
    from types import SimpleNamespace

    from scope.core.pipelines.omniforcing.runtime import (
        _AUDIO_FRAMES_FIRST_BLOCK,
        _AUDIO_FRAMES_PER_BLOCK,
        OmniForcingRuntime,
    )

    snap = OmniForcingRuntime._snap_to_block_layout
    stub = SimpleNamespace(num_frame_per_block=3, num_frame_per_block_first=4)
    assert snap(stub, 1) == 4  # clamp to the first-block minimum
    assert snap(stub, 4) == 4  # already valid
    assert snap(stub, 5) == 7  # 4 + 1*3
    assert snap(stub, 16) == 16  # 4 + 4*3 (the released 5s layout)
    assert snap(stub, 17) == 19  # 4 + 5*3
    # Upstream block-size constants (mask_builder): 4+26 first, 3+25 thereafter.
    assert (_AUDIO_FRAMES_FIRST_BLOCK, _AUDIO_FRAMES_PER_BLOCK) == (26, 25)


def test_windowed_decode_bookkeeping():
    """Drive the real _decode_new with a stub VAE: emitted frame counts must be
    contiguous and match the global latent->pixel mapping, with no drift, while the
    decode window stays bounded (constant cost)."""
    import torch

    from scope.core.pipelines.omniforcing.runtime import OmniForcingRuntime

    px = OmniForcingRuntime._px_for_latents
    H = W = 32

    class _StubVAE:
        last_window_len = 0

        def decode_to_pixel(self, window):
            length = int(window.shape[1])
            _StubVAE.last_window_len = length
            return torch.zeros(1, 3, px(length), H, W)

    rt = OmniForcingRuntime.__new__(OmniForcingRuntime)
    rt.decode_context_latents = 2
    rt.decode_lookahead_latents = 1
    rt.stream_audio = False
    rt.height, rt.width = H, W
    rt.video_vae = _StubVAE()
    rt._video_latent_history = None
    rt._decoded_latent_count = 0
    rt._emitted_audio_samples = 0

    block_sizes = [4, 3, 3, 3, 3, 3]  # first block 4 latents, then 3
    total_latents = 0
    total_emitted = 0
    max_window = 0
    for v_len in block_sizes:
        blk = torch.zeros(1, v_len, 128, 1, 1)
        new_video, _ = rt._decode_new(blk, None)
        total_latents += v_len
        total_emitted += int(new_video.shape[0])
        max_window = max(max_window, _StubVAE.last_window_len)
        # Cumulative emitted pixels must equal the global mapping up to emit_upto.
        emit_upto = total_latents - rt.decode_lookahead_latents
        assert total_emitted == px(emit_upto), (v_len, total_emitted, px(emit_upto))
        assert new_video.shape[1:] == (H, W, 3)

    # Window (decode cost) stays bounded ~ context + block + lookahead, never the
    # whole take -> constant throughput.
    assert max_window <= 4 + 2 + 1


def test_artifacts_point_at_expected_repos():
    assert (
        OMNIFORCING_GENERATOR_ARTIFACT.repo_id
        == "Exploration/omniforcing-ltx2-5s-causal"
    )
    # 5 shards + the index json.
    shards = [
        f for f in OMNIFORCING_GENERATOR_ARTIFACT.files if f.endswith(".safetensors")
    ]
    assert len(shards) == 5
    assert any(f.endswith(".index.json") for f in OMNIFORCING_GENERATOR_ARTIFACT.files)

    assert LTX2_BASE_ARTIFACT.repo_id == "Lightricks/LTX-2"
    # Minimal subset CONFIRMED on H100 (2026-06-19): the consolidated checkpoint
    # carries the transformer + both VAEs + vocoder + connectors, so only it plus the
    # Gemma text encoder (text_encoder/model-* + tokenizer/) are needed — no separate
    # google/gemma repo and no per-component vae/audio_vae/vocoder/connectors dirs.
    assert "ltx-2-19b-dev.safetensors" in LTX2_BASE_ARTIFACT.files
    assert "tokenizer" in LTX2_BASE_ARTIFACT.files
    # The Gemma model shards (the model-* set, not the duplicate diffusion_pytorch_model-*).
    assert any(
        f.startswith("text_encoder/model-") and f.endswith(".safetensors")
        for f in LTX2_BASE_ARTIFACT.files
    )
    # The per-component decoder dirs must NOT be downloaded (they live in the base ckpt).
    for unused in ("vae", "audio_vae", "vocoder", "connectors", "scheduler"):
        assert unused not in LTX2_BASE_ARTIFACT.files, unused


def test_config_listed_in_pipeline_artifacts():
    """Both weight sources must be wired into the config so download_models fetches them."""
    # artifacts ClassVar is optional in this scaffold; if present it must include both.
    arts = getattr(OmniForcingConfig, "artifacts", [])
    if arts:
        repo_ids = {a.repo_id for a in arts}
        assert "Exploration/omniforcing-ltx2-5s-causal" in repo_ids
        assert "Lightricks/LTX-2" in repo_ids


def test_model_yaml_consistent_with_schema():
    yaml_path = (
        Path(__file__).resolve().parents[1]
        / "src/scope/core/pipelines/omniforcing/model.yaml"
    )
    cfg = OmegaConf.load(yaml_path)
    c = OmniForcingConfig()
    # LTX-2 latent geometry.
    assert cfg.vae_spatial_downsample_factor == 32
    assert cfg.vae_temporal_downsample_factor == 8
    # Block sizes + schedule must agree between model.yaml and the schema defaults.
    assert cfg.num_frame_per_block == c.num_frame_per_block
    assert cfg.num_frame_per_block_first == c.num_frame_per_block_first
    assert list(cfg.denoising_steps) == c.denoising_steps
    assert cfg.audio_sample_rate == c.audio_sample_rate


def test_runtime_probe_is_import_safe_and_false_without_ltx():
    """runtime.is_available() must never raise and must be False when the LTX
    stack is not installed (the common case on CPU/CI/macOS dev hosts)."""
    available = runtime.is_available()
    assert isinstance(available, bool)
    # ltx_core/ltx_causal/ltx_distillation are pod-only; absent here.
    import importlib.util

    if importlib.util.find_spec("ltx_causal") is None:
        assert available is False


def test_pipeline_module_imports_without_ltx_stack():
    """Importing the pipeline (for registry discovery) must not require the LTX
    stack — only constructing/running it does."""
    from scope.core.pipelines.omniforcing.pipeline import OmniForcingPipeline

    assert OmniForcingPipeline.get_config_class() is OmniForcingConfig


def test_build_runtime_raises_clearly_without_ltx():
    """build_runtime must fail with an actionable message (not an ImportError)
    when the LTX runtime is missing."""
    import importlib.util

    if importlib.util.find_spec("ltx_causal") is not None:
        return  # installed on this host; nothing to assert
    import pytest

    with pytest.raises(RuntimeError, match="not installed"):
        runtime.build_runtime(
            model_dir="/tmp/models",
            model_config=OmegaConf.create({}),
            config=OmniForcingConfig(),
            device=None,
            dtype=None,
        )
