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
    # The base repo must supply the consolidated checkpoint + every decoder/encoder
    # component OmniForcing needs (so no separate gated google/gemma repo).
    for needed in (
        "ltx-2-19b-dev.safetensors",
        "vae",
        "audio_vae",
        "vocoder",
        "text_encoder",
        "tokenizer",
    ):
        assert needed in LTX2_BASE_ARTIFACT.files, needed


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
