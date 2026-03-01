"""
IP-Adapter Encoding Block for face identity conditioning.

Handles face detection, alignment, and ArcFace embedding extraction from a
reference face image. Produces ip_image_embed that flows through the pipeline
to IPAdapterWanModel's Perceiver Resampler → IP cross-attention layers.

The face embedding is computed once and cached — it doesn't change per chunk,
so it provides constant identity anchoring without per-chunk cost.
"""

import logging
from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    InputParam,
    OutputParam,
)

logger = logging.getLogger(__name__)


class IPAdapterEncodingBlock(ModularPipelineBlocks):
    """Encodes a face reference image into an ArcFace embedding for IP-Adapter.

    Input: ip_face_image (path to face image)
    Output: ip_image_embed [1, 1, 512] — ArcFace embedding ready for Resampler

    The embedding is cached after first computation. Passing a new ip_face_image
    recomputes it. Passing None clears it.
    """

    def __init__(self):
        super().__init__()
        self._face_analyzer = None
        self._face_encoder = None
        self._cached_embed = None
        self._cached_image_path = None

    @property
    def expected_components(self) -> list:
        return []

    @property
    def expected_configs(self) -> list:
        return []

    @property
    def description(self) -> str:
        return "IPAdapterEncodingBlock: Encode face reference image into ArcFace embedding for IP-Adapter identity conditioning"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "ip_face_image",
                default=None,
                type_hint=str | None,
                description="Path to face reference image for identity conditioning. Cached after first encode.",
            ),
            InputParam(
                "ip_scale",
                default=1.0,
                type_hint=float,
                description="Scaling factor for IP-Adapter identity injection (0.0 to 2.0)",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "ip_image_embed",
                type_hint=torch.Tensor | None,
                description="ArcFace face embedding [1, 1, 512] for IP-Adapter",
            ),
            OutputParam(
                "ip_scale",
                type_hint=float,
                description="IP-Adapter injection scale",
            ),
        ]

    def _ensure_face_models(self, device):
        """Lazily load face detection and recognition models."""
        if self._face_analyzer is not None:
            return

        try:
            from insightface.app import FaceAnalysis

            self._face_analyzer = FaceAnalysis(
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self._face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        except ImportError:
            raise ImportError(
                "insightface is required for IP-Adapter face encoding. "
                "Install it with: pip install insightface onnxruntime-gpu"
            )

        try:
            from facexlib.recognition import init_recognition_model

            self._face_encoder = init_recognition_model("arcface", device=device)
            self._face_encoder.eval()
        except ImportError:
            raise ImportError(
                "facexlib is required for IP-Adapter face encoding. "
                "Install it with: pip install facexlib"
            )

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        ip_face_image = block_state.ip_face_image
        ip_scale = getattr(block_state, "ip_scale", 1.0)

        # No face image — clear embedding
        if ip_face_image is None:
            block_state.ip_image_embed = None
            block_state.ip_scale = ip_scale
            self.set_block_state(state, block_state)
            return components, state

        # Return cached embedding if same image
        if self._cached_embed is not None and self._cached_image_path == ip_face_image:
            block_state.ip_image_embed = self._cached_embed
            block_state.ip_scale = ip_scale
            self.set_block_state(state, block_state)
            return components, state

        # Load face models lazily
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ensure_face_models(device)

        # Load and detect face
        import cv2
        from insightface.utils import face_align

        img = cv2.imread(ip_face_image)
        if img is None:
            logger.warning(
                f"IPAdapterEncodingBlock: Could not load image {ip_face_image}"
            )
            block_state.ip_image_embed = None
            block_state.ip_scale = ip_scale
            self.set_block_state(state, block_state)
            return components, state

        # Detect faces
        faces = self._face_analyzer.get(img)
        if len(faces) == 0:
            logger.warning(
                f"IPAdapterEncodingBlock: No face detected in {ip_face_image}"
            )
            block_state.ip_image_embed = None
            block_state.ip_scale = ip_scale
            self.set_block_state(state, block_state)
            return components, state

        # Pick largest face
        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        # Align and crop to 112x112
        aligned = face_align.norm_crop(img, landmark=face.kps, image_size=112)

        # Convert to tensor: [1, 3, 112, 112], normalized to [-1, 1]
        import torchvision.transforms as T

        transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        face_tensor = transform(aligned).unsqueeze(0).to(device)

        # ArcFace expects BGR
        face_tensor = face_tensor[:, [2, 1, 0], :, :].contiguous()

        # Extract embedding
        embedding = self._face_encoder(face_tensor)  # [1, 512], L2-normalized

        # Reshape to [1, 1, 512] for Resampler input
        ip_image_embed = embedding.reshape(1, 1, 512)

        # Cache for subsequent chunks
        self._cached_embed = ip_image_embed
        self._cached_image_path = ip_face_image

        logger.info(
            f"IPAdapterEncodingBlock: Encoded face from {ip_face_image}, "
            f"embedding shape={ip_image_embed.shape}"
        )

        block_state.ip_image_embed = ip_image_embed
        block_state.ip_scale = ip_scale
        self.set_block_state(state, block_state)
        return components, state
