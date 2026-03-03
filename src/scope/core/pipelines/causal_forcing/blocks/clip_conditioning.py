"""CLIP image conditioning block for I2V generation.

Extracts CLIP visual features from a reference image and stores them in state
for use by the generator model's cross-attention conditioning (via clip_fea parameter).
"""

from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)


class CLIPImageConditioningBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("clip_encoder", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Extract CLIP visual features from reference image for I2V conditioning"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "reference_image_tensor",
                type_hint=torch.Tensor | None,
                description=(
                    "Reference image tensor in [-1, 1] range with shape [B, C, H, W] "
                    "or [B, C, F, H, W]. Set by pipeline when reference_image is provided."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "clip_fea",
                type_hint=torch.Tensor | None,
                description="CLIP visual features [B, 257, 1280] for generator cross-attention",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        # Only use CLIP conditioning if reference image is provided
        # clip_encoder component will only exist when pipeline is initialized with reference_image
        if block_state.reference_image_tensor is not None:
            if not hasattr(components, "clip_encoder"):
                raise RuntimeError(
                    "CLIP encoder required for I2V mode but not loaded. "
                    "This indicates a pipeline initialization error."
                )
            clip_fea = components.clip_encoder(block_state.reference_image_tensor)
            block_state.clip_fea = clip_fea
        else:
            block_state.clip_fea = None

        self.set_block_state(state, block_state)
        return components, state
