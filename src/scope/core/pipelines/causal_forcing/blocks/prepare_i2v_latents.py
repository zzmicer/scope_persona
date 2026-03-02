"""I2V latent preparation block for Causal-Forcing.

Encodes a reference image through the VAE to create an initial latent frame,
then generates noise for the remaining frames. The initial latent is used as
the conditioning signal (y parameter) for the generator's I2V mode.
"""

from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)


class PrepareI2VLatentsBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
            ComponentSpec("vae", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 1),
            ConfigSpec("vae_spatial_downsample_factor", 8),
        ]

    @property
    def description(self) -> str:
        return (
            "Prepare latents for I2V mode: encode reference image via VAE to create "
            "initial latent, then generate noise for remaining frames"
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "reference_image_tensor",
                required=True,
                type_hint=torch.Tensor,
                description="Reference image tensor [B, C, H, W] or [B, C, F, H, W] in [-1, 1]",
            ),
            InputParam(
                "base_seed",
                type_hint=int,
                default=42,
                description="Base seed for random number generation",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index",
            ),
            InputParam(
                "height",
                type_hint=int,
                description="Output height in pixels",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="Output width in pixels",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Noisy latents to denoise [B, F, C, H_lat, W_lat]",
            ),
            OutputParam(
                "i2v_initial_latent",
                type_hint=torch.Tensor | None,
                description="VAE-encoded initial latent for I2V conditioning [B, 1, C, H_lat, W_lat]",
            ),
            OutputParam("generator", description="Random number generator"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        generator_param = next(components.generator.model.parameters())
        device = generator_param.device
        dtype = generator_param.dtype

        latent_height = (
            block_state.height // components.config.vae_spatial_downsample_factor
        )
        latent_width = (
            block_state.width // components.config.vae_spatial_downsample_factor
        )

        base_seed = block_state.base_seed
        if base_seed is None:
            base_seed = 42

        block_seed = base_seed + block_state.current_start_frame
        rng = torch.Generator(device=device).manual_seed(block_seed)

        num_latent_frames = components.config.num_frame_per_block
        if block_state.current_start_frame == 0 and num_latent_frames < 2:
            num_latent_frames = 2

        ref_image = block_state.reference_image_tensor

        if block_state.current_start_frame == 0:
            # First chunk: encode reference image as initial latent
            # Ensure image is in video format [B, C, F, H, W]
            if ref_image.ndim == 4:
                ref_image = ref_image.unsqueeze(2)  # [B, C, 1, H, W]

            # Resize to match output resolution if needed
            if (
                ref_image.shape[-2] != block_state.height
                or ref_image.shape[-1] != block_state.width
            ):
                ref_image = torch.nn.functional.interpolate(
                    ref_image.squeeze(2),
                    size=(block_state.height, block_state.width),
                    mode="bicubic",
                    align_corners=False,
                ).unsqueeze(2)

            ref_image = ref_image.to(device=device, dtype=dtype)

            # Encode through VAE (use_cache=False for one-time encode)
            initial_latent = components.vae.encode_to_latent(
                ref_image, use_cache=False
            )  # [B, 1, C, H_lat, W_lat]

            block_state.i2v_initial_latent = initial_latent

            # Generate noise for remaining frames
            noise_frames = num_latent_frames
            latents = torch.randn(
                [1, noise_frames, 16, latent_height, latent_width],
                device=device,
                dtype=dtype,
                generator=rng,
            )
        else:
            # Subsequent chunks: standard noise generation, no initial latent
            block_state.i2v_initial_latent = None
            latents = torch.randn(
                [1, num_latent_frames, 16, latent_height, latent_width],
                device=device,
                dtype=dtype,
                generator=rng,
            )

        block_state.latents = latents
        block_state.generator = rng

        self.set_block_state(state, block_state)
        return components, state
