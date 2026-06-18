# SPDX-License-Identifier: Apache-2.0
"""Wan2.2 VAE (48-channel latents, 16x spatial / 4x temporal compression).

Provides a ``create_vae`` factory matching the signature of
``scope.core.pipelines.wan2_1.vae.create_vae`` so the LongLive pipeline can swap
VAEs without changing call sites.

Upstream reference: NVlabs/LongLive ``wan_5b/modules/vae2_2.py`` (full Wan2.2
VAE). An optional ``lightvae`` variant maps to the LightVAE-5B checkpoint
(Skywork/Matrix-Game-3.0, ``MG-LightVAE.pth``); the full Wan2.2 VAE is default.

Usage:
    from scope.core.pipelines.wan2_2.vae import create_vae

    # Default (full Wan2.2 VAE)
    vae = create_vae(model_dir="wan_models")

    # Explicit path override
    vae = create_vae(model_dir="wan_models", vae_path="/path/to/Wan2.2_VAE.pth")

    # LightVAE-5B variant (hook; checkpoint must be provided)
    vae = create_vae(model_dir="wan_models", vae_type="lightvae")
"""

from functools import partial

from .constants import WAN2_2_VAE_LATENT_MEAN, WAN2_2_VAE_LATENT_STD
from .vae2_2 import Wan2_2_VAEWrapper

# UI/dropdown registry, mirroring wan2_1/vae/__init__.py's VAE_REGISTRY shape.
VAE_REGISTRY: dict[str, object] = {
    "wan2_2": Wan2_2_VAEWrapper,
    "lightvae": partial(Wan2_2_VAEWrapper, use_lightvae=True),
}

DEFAULT_VAE_TYPE = "wan2_2"


def create_vae(
    model_dir: str = "wan_models",
    model_name: str | None = None,
    vae_type: str | None = None,
    vae_path: str | None = None,
) -> Wan2_2_VAEWrapper:
    """Create a Wan2.2 VAE instance by type.

    Signature-compatible with ``wan2_1.vae.create_vae``.

    Args:
        model_dir: Base model directory.
        model_name: Model subdirectory name. Defaults to "Wan2.2-TI2V-5B".
        vae_type: VAE type ("wan2_2" for the full 48-ch VAE, "lightvae" for the
            optional LightVAE-5B variant). Defaults to "wan2_2".
        vae_path: Optional explicit checkpoint path override.

    Returns:
        Initialized ``Wan2_2_VAEWrapper`` instance.

    Raises:
        ValueError: If ``vae_type`` is not recognized.
    """
    vae_type = vae_type or DEFAULT_VAE_TYPE
    if model_name is None:
        model_name = "Wan2.2-TI2V-5B"

    vae_factory = VAE_REGISTRY.get(vae_type)
    if vae_factory is None:
        available = list(VAE_REGISTRY.keys())
        raise ValueError(
            f"create_vae: Unknown VAE type '{vae_type}'. Available types: {available}"
        )

    return vae_factory(model_dir=model_dir, model_name=model_name, vae_path=vae_path)


__all__ = [
    "Wan2_2_VAEWrapper",
    "create_vae",
    "VAE_REGISTRY",
    "DEFAULT_VAE_TYPE",
    "WAN2_2_VAE_LATENT_MEAN",
    "WAN2_2_VAE_LATENT_STD",
]
