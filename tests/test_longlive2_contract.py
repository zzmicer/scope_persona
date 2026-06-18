"""CPU-runnable contract tests for the LongLive 2.0 (Wan2.2-TI2V-5B) pipeline.

These do NOT load the multi-GB checkpoints or require a GPU. They lock the
architectural + wiring invariants that the longlive2 port depends on, so that
the kinds of regressions that previously only surfaced as "renders pure noise"
or "loads the wrong checkpoint" fail fast in CI instead:

  * the 5B transformer config (channels / dims / heads / layers),
  * the RoPE head-dim split summing back to head_dim,
  * the distilled denoising-schedule subsample (e.g. 2-step -> [1000, 681]),
  * the NVFP4 backend -> checkpoint-file selection (FourOverSix vs TE),
  * model.yaml staying consistent with the 48-channel Wan2.2 VAE,
  * the NVFP4 runtime probe being import-safe (returns False on CPU).
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from scope.core.pipelines.longlive2.pipeline import LongLive2Pipeline
from scope.core.pipelines.longlive2.schema import (
    PRECISION_ARTIFACTS,
    PRECISION_STEPS,
    LongLive2Config,
)


def _import_causal_model():
    """Import CausalWanModel or skip if pod-only deps (flash_attn) are absent.

    The 5B transformer transitively imports the wan2_1 attention module, which
    requires flash_attn — present on the Blackwell pod but not on CPU/macOS dev
    boxes. These tests are meaningful only where the model can be built, so skip
    (don't fail) when the dependency is unavailable.
    """
    try:
        from scope.core.pipelines.wan2_2.modules.causal_model import CausalWanModel
    except ModuleNotFoundError as exc:  # e.g. flash_attn / triton on CPU hosts
        pytest.skip(f"CausalWanModel deps unavailable on this host: {exc}")
    return CausalWanModel


def _causal_model_init_defaults() -> dict:
    CausalWanModel = _import_causal_model()

    sig = inspect.signature(CausalWanModel.__init__)
    return {
        name: p.default
        for name, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def test_causal_model_defaults_match_5b_spec():
    """The CausalWanModel defaults must match Wan2.2-TI2V-5B, not the 1.3B model."""
    d = _causal_model_init_defaults()
    assert d["model_type"] == "ti2v"
    assert d["in_dim"] == 48, "Wan2.2 VAE has 48 latent channels"
    assert d["out_dim"] == 48
    assert d["dim"] == 3072
    assert d["ffn_dim"] == 14336
    assert d["num_heads"] == 24
    assert d["num_layers"] == 30
    assert d["freq_dim"] == 256
    assert d["eps"] == 1e-6
    assert d["qk_norm"] is True
    assert d["cross_attn_norm"] is True
    # head_dim must be 128 (3072 / 24) and even (RoPE requires it).
    assert d["dim"] % d["num_heads"] == 0
    assert (d["dim"] // d["num_heads"]) == 128


def test_causal_model_rope_freqs_sum_to_head_dim():
    """A tiny instantiation verifies the parametric RoPE split is consistent.

    The freq table is cat([rope(d-4*(d//6)), rope(2*(d//6)), rope(2*(d//6))]),
    each rope() producing dim/2 complex entries, so the last dim must be
    head_dim // 2 for ANY head_dim. Building a tiny model (not the 5B weights)
    is enough to catch a broken split.
    """
    CausalWanModel = _import_causal_model()

    model = CausalWanModel(
        model_type="ti2v",
        in_dim=48,
        out_dim=48,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        text_dim=32,
        freq_dim=64,
    )
    head_dim = model.dim // model.num_heads
    assert head_dim == 16
    assert torch.is_complex(model.freqs)
    assert model.freqs.shape[1] == head_dim // 2


@pytest.mark.parametrize(
    "num_steps,expected",
    [
        (4, [1000, 946, 854, 681]),  # full schedule
        (2, [1000, 681]),  # nvfp4-s2 heuristic subsample
        (1, [1000]),  # degenerate single step
        (8, [1000, 946, 854, 681]),  # more steps than the schedule -> unchanged
    ],
)
def test_subsample_schedule(num_steps, expected):
    base = [1000, 946, 854, 681]
    assert LongLive2Pipeline._subsample_schedule(base, num_steps) == expected


def test_resolve_steps_matches_precision():
    assert LongLive2Config(precision="nvfp4-s2").resolve_steps() == 2
    assert LongLive2Config(precision="nvfp4-s4").resolve_steps() == 4
    assert LongLive2Config(precision="bf16").resolve_steps() == 4
    # explicit override wins
    assert LongLive2Config(precision="nvfp4-s2", steps=3).resolve_steps() == 3


def test_precision_tables_consistent():
    assert set(PRECISION_STEPS) == set(PRECISION_ARTIFACTS)
    assert PRECISION_ARTIFACTS["nvfp4-s2"].repo_id.endswith("NVFP4-S2")
    assert PRECISION_ARTIFACTS["nvfp4-s4"].repo_id.endswith("NVFP4-S4")


@pytest.mark.parametrize(
    "use_te,expected_file",
    [(False, "model_4o6.pt"), (True, "model_te.pt")],
)
def test_inject_nvfp4_config_selects_backend_checkpoint(use_te, expected_file):
    """_inject_nvfp4_config must resolve the checkpoint that matches the backend.

    This is the contract the manager (generator_path) has to agree with: default
    FourOverSix -> model_4o6.pt, TransformerEngine -> model_te.pt. It must also
    set model_quant=True and the 'mse' scale rules the released weights use.
    """
    model_config = OmegaConf.create({})
    config = SimpleNamespace(model_quant_use_transformer_engine=use_te)
    LongLive2Pipeline._inject_nvfp4_config(
        model_config, config, "nvfp4-s2", model_dir="/tmp/models"
    )
    assert model_config.model_quant is True
    assert model_config.model_quant_use_transformer_engine is use_te
    assert model_config.generator_ckpt.endswith(
        f"LongLive-2.0-5B-NVFP4-S2/{expected_file}"
    )
    assert model_config.model_quant_scale_rule == "mse"


def test_inject_nvfp4_config_requires_model_dir():
    with pytest.raises(ValueError):
        LongLive2Pipeline._inject_nvfp4_config(
            OmegaConf.create({}), SimpleNamespace(), "nvfp4-s2", model_dir=None
        )


def test_model_yaml_consistent_with_wan22_vae():
    """model.yaml must describe the 48-channel / 16x Wan2.2 VAE and the verified
    4-step distilled schedule, since the shared blocks read these from config."""
    yaml_path = (
        Path(__file__).resolve().parents[1]
        / "src/scope/core/pipelines/longlive2/model.yaml"
    )
    cfg = OmegaConf.load(yaml_path)
    assert cfg.latent_channels == 48
    assert cfg.vae_spatial_downsample_factor == 16
    assert cfg.vae_temporal_downsample_factor == 4
    assert list(cfg.denoising_steps) == [1000, 946, 854, 681]


def test_nvfp4_available_is_import_safe_and_false_on_cpu():
    """The NVFP4 probe must never raise and must be False without CUDA."""
    from scope.core.pipelines.wan2_2.nvfp4 import nvfp4_available, nvfp4_capabilities

    caps = nvfp4_capabilities()
    assert set(caps) >= {"cuda", "fouroversix", "transformer_engine", "triton"}
    available = nvfp4_available()
    assert isinstance(available, bool)
    if not torch.cuda.is_available():
        assert available is False
