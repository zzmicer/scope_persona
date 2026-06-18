"""OmniForcing / LTX-2 runtime adapter.

This module isolates every import of the external OmniForcing LTX packages
(``ltx_core``, ``ltx_causal``, ``ltx_distillation``) behind lazy functions so that
``scope.core.pipelines.omniforcing`` is import-safe on hosts that do not have the
LTX stack installed (e.g. macOS / CI) — the registry must be able to import the
pipeline module to read its config, and the contract tests run on CPU.

The LTX packages are NOT vendored into this repo (LTX-2 Community License +
~tens of thousands of lines). They are installed on the GPU host per
``docs/usage.md``. All OmniForcing-specific construction and the autoregressive
audio-video inference loop live here, behind ``build_runtime``.

Upstream references:
  - generator + base load: omniforcing_causal_inference.py
  - causal AR loop: ltx_distillation/inference/causal_pipeline.py
    (CausalAVInferencePipeline)
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Modules that must be importable for the real (GPU) path to run.
_REQUIRED_MODULES = ("ltx_core", "ltx_causal", "ltx_distillation")


def is_available() -> bool:
    """Return True if the OmniForcing LTX runtime is installed on this host.

    Cheap check (does not import torch-heavy modules): only inspects whether the
    package specs can be found. The actual heavy imports happen in
    ``build_runtime``.
    """
    try:
        return all(
            importlib.util.find_spec(name) is not None for name in _REQUIRED_MODULES
        )
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def _resolve_paths(model_dir: str, model_config: Any) -> dict[str, Path]:
    """Resolve on-disk weight paths under ``model_dir``.

    Layout mirrors how scope downloads HF artifacts: ``<model_dir>/<repo-last-
    segment>/<files>``. The OmniForcing generator repo and the Lightricks/LTX-2
    base repo each get their own subdirectory.
    """
    root = Path(model_dir)
    base_dir = root / "LTX-2"
    gen_dir = root / "omniforcing-ltx2-5s-causal"
    base_ckpt = getattr(model_config, "base_checkpoint", "ltx-2-19b-dev.safetensors")
    return {
        "base_checkpoint": base_dir / base_ckpt,
        "generator_index": gen_dir
        / "omniforcing_ltx2_5s_causal.safetensors.index.json",
        "generator_dir": gen_dir,
        # Gemma-3-12B is bundled in the LTX-2 repo (transformers layout).
        "gemma": base_dir / "text_encoder",
        "tokenizer": base_dir / "tokenizer",
        "vae": base_dir / "vae",
        "audio_vae": base_dir / "audio_vae",
        "vocoder": base_dir / "vocoder",
    }


def build_runtime(
    *,
    model_dir: str,
    model_config: Any,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> OmniForcingRuntime:
    """Construct the full OmniForcing AV runtime (GPU host only).

    Loads the causal generator (base LTX-2 weights + distilled OmniForcing
    weights), the Gemma text encoder, the video + audio VAEs and the vocoder, and
    wraps them in the autoregressive ``CausalAVInferencePipeline``.

    Raises ``RuntimeError`` if the LTX runtime is not installed. The exact loader
    entry points are confirmed against the installed package on the pod (see
    docs/usage.md); the imports below name the upstream modules.
    """
    if not is_available():
        raise RuntimeError(
            "OmniForcing LTX runtime is not installed. Install ltx-core, "
            "ltx-causal and ltx-distillation on the GPU host (see "
            "src/scope/core/pipelines/omniforcing/docs/usage.md)."
        )
    # Heavy imports are deferred to here so module import stays cheap/safe.
    return OmniForcingRuntime(
        paths=_resolve_paths(model_dir, model_config),
        model_config=model_config,
        config=config,
        device=device,
        dtype=dtype,
    )


class OmniForcingRuntime:
    """Holds the loaded LTX-2 components + the causal AV inference pipeline.

    Constructed only via :func:`build_runtime` on a GPU host. The body performs
    the OmniForcing model loading and exposes :meth:`generate_chunk`, which runs
    one autoregressive pass and returns decoded audio + video for that chunk.
    """

    def __init__(
        self,
        *,
        paths: dict[str, Path],
        model_config: Any,
        config: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        # Imported lazily (only reachable on an LTX-installed host).
        from ltx_causal.transformer.causal_model import CausalLTXModel  # noqa: F401
        from ltx_distillation.inference.causal_pipeline import (
            CausalAVInferencePipeline,  # noqa: F401
        )
        from ltx_distillation.models.text_encoder_wrapper import (
            create_text_encoder_wrapper,  # noqa: F401
        )

        self.paths = paths
        self.model_config = model_config
        self.config = config
        self.device = device
        self.dtype = dtype

        # ------------------------------------------------------------------
        # Model loading + pipeline construction.
        #
        # This is the scaffold seam: the precise constructor signatures
        # (CausalLTXModel config, base-checkpoint load that strips audio tokens,
        # generator-weight load, add_noise_fn, denoising_sigmas conversion from
        # the timestep schedule) are finalized against the installed package on
        # the H100/H200 pod. See omniforcing_causal_inference.py for the upstream
        # construction sequence we mirror.
        # ------------------------------------------------------------------
        raise NotImplementedError(
            "OmniForcing GPU runtime wiring is finalized on the pod — see "
            "docs/usage.md (Phase: pod bring-up). The scope-side scaffold "
            "(schema/artifacts/registry/contract tests) is complete; this body "
            "loads CausalLTXModel + VAEs + Gemma and builds "
            "CausalAVInferencePipeline against the installed ltx-distillation."
        )

    def generate_chunk(
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        init_cache: bool = False,
    ) -> dict[str, Any]:
        """Run one autoregressive AV pass and return decoded video + audio.

        Returns the scope AV-dict shape (see pipeline.OmniForcingPipeline):
        ``{video, video_timestamps, audio, audio_sample_rate, audio_timestamps,
        frame_rate}``.
        """
        raise NotImplementedError  # pragma: no cover - finalized on pod
