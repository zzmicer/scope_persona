"""Helios Distilled autoregressive video generation pipeline for Daydream Scope.

Uses the HuggingFace diffusers HeliosPyramidPipeline to load and run the
Helios-Distilled model. Components (transformer, VAE, scheduler) are loaded
individually from the local models directory to avoid issues with
model_index.json auto-detection.

The model generates video autoregressively in 33-frame chunks via pyramid
multi-scale denoising with multi-term memory patchification.
"""

import gc
import logging
import time
from typing import TYPE_CHECKING

import torch

from scope.core.config import get_model_file_path
from scope.core.pipelines.interface import Pipeline

from .schema import HeliosConfig

if TYPE_CHECKING:
    from scope.core.pipelines.schema import BasePipelineConfig

logger = logging.getLogger(__name__)

MODEL_DIR_NAME = "Helios-Distilled"


class HeliosPipeline(Pipeline):
    """Helios Distilled video generation pipeline.

    Wraps the diffusers HeliosPyramidPipeline for use within Daydream Scope.
    The model generates video autoregressively — each chunk produces 33 pixel
    frames (9 latent frames) using pyramid multi-scale denoising with only
    2 steps per stage.

    Diffusers provides two Helios pipeline classes:
        - HeliosPipeline: stage1 only, for Helios-Base
        - HeliosPyramidPipeline: pyramid multi-scale, for Helios-Mid/Distilled

    This plugin uses HeliosPyramidPipeline with HeliosDMDScheduler.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return HeliosConfig

    def __init__(
        self,
        height: int = 384,
        width: int = 640,
        num_frames: int = 240,
        pyramid_steps: int = 2,
        amplify_first_chunk: bool = True,
        offload_text_encoder: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.dtype = dtype
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.pyramid_steps = pyramid_steps
        self.amplify_first_chunk = amplify_first_chunk
        self.offload_text_encoder = offload_text_encoder

        if kwargs:
            logger.debug(f"HeliosPipeline ignoring unknown kwargs: {list(kwargs.keys())}")

        start = time.time()
        model_path = str(get_model_file_path(MODEL_DIR_NAME))
        logger.info(f"Loading Helios Distilled from: {model_path}")

        self._pipe = self._load_pipeline(model_path)

        logger.info(f"Helios Distilled loaded in {time.time() - start:.2f}s")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory after load: {allocated:.2f} GB")

        self._cached_prompt_text: str | None = None

    def _load_pipeline(self, model_path: str):
        """Load the Helios diffusers pipeline with explicit component loading.

        Loads transformer, VAE, and scheduler individually from subfolder paths,
        then assembles them into a HeliosPyramidPipeline. This avoids relying on
        model_index.json auto-detection which can fail if the installed diffusers
        version doesn't register the class name properly.

        Helios-Distilled uses HeliosPyramidPipeline (pyramid multi-scale
        denoising) — NOT HeliosPipeline (stage1 only, for the Base model).
        """
        from diffusers import HeliosPyramidPipeline as DiffusersHeliosPyramidPipeline
        from diffusers.models import AutoencoderKLWan, HeliosTransformer3DModel
        from diffusers.schedulers import HeliosDMDScheduler

        logger.info("  Loading transformer...")
        transformer = HeliosTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
        )

        logger.info("  Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        logger.info("  Loading scheduler...")
        scheduler = HeliosDMDScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
        )

        logger.info("  Assembling HeliosPyramidPipeline...")
        pipe = DiffusersHeliosPyramidPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            torch_dtype=self.dtype,
        )

        pipe.to(self.device)

        if self.offload_text_encoder:
            self._offload_text_encoder_from_pipe(pipe)

        return pipe

    def _offload_text_encoder_from_pipe(self, pipe) -> None:
        """Move text encoder to CPU to free VRAM."""
        try:
            if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                pipe.text_encoder.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Text encoder offloaded to CPU")
        except Exception as e:
            logger.warning(f"Failed to offload text encoder: {e}")

    def _offload_text_encoder(self) -> None:
        """Move text encoder to CPU to free VRAM."""
        self._offload_text_encoder_from_pipe(self._pipe)

    def _ensure_text_encoder_on_device(self) -> None:
        """Move text encoder back to GPU if it was offloaded."""
        if not self.offload_text_encoder:
            return
        try:
            if hasattr(self._pipe, "text_encoder") and self._pipe.text_encoder is not None:
                if next(self._pipe.text_encoder.parameters()).device.type == "cpu":
                    self._pipe.text_encoder.to(self.device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    logger.info("Text encoder moved back to GPU")
        except Exception as e:
            logger.warning(f"Failed to move text encoder to device: {e}")

    def __call__(self, **kwargs) -> dict:
        """Generate video from text prompt.

        Args:
            **kwargs: Generation parameters including:
                - prompts: List of prompt dicts with 'text' and 'weight' keys
                - base_seed: Random seed for generation
                - num_frames: Override frame count
                - height/width: Override resolution
                - pyramid_steps: Override denoising steps per stage
                - amplify_first_chunk: Override first chunk amplification

        Returns:
            Dictionary with "video" key containing [T, H, W, C] tensor in [0, 1] range.
        """
        return self._generate(**kwargs)

    @torch.inference_mode()
    def _generate(self, **kwargs) -> dict:
        prompts = kwargs.get("prompts", [{"text": "a beautiful landscape", "weight": 1.0}])
        seed = kwargs.get("base_seed", kwargs.get("seed", 42))
        num_frames = kwargs.get("num_frames", self.num_frames)
        height = kwargs.get("height", self.height)
        width = kwargs.get("width", self.width)
        pyramid_steps = kwargs.get("pyramid_steps", self.pyramid_steps)
        amplify_first_chunk = kwargs.get("amplify_first_chunk", self.amplify_first_chunk)

        prompt_text = prompts[0]["text"] if prompts else "a beautiful landscape"

        prompt_changed = prompt_text != self._cached_prompt_text
        if prompt_changed:
            self._ensure_text_encoder_on_device()

        steps_list = [pyramid_steps] * 3

        logger.info(
            f"Generating {num_frames} frames at {height}x{width}, "
            f"pyramid_steps={steps_list}, seed={seed}"
        )

        gen_start = time.time()

        # HeliosPyramidPipeline: always pyramid/stage2, guidance_scale=1.0 for distilled
        output = self._pipe(
            prompt=prompt_text,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=1.0,
            pyramid_num_inference_steps_list=steps_list,
            is_amplify_first_chunk=amplify_first_chunk,
            generator=torch.Generator(self.device).manual_seed(seed),
        )

        gen_time = time.time() - gen_start
        logger.info(f"Generation completed in {gen_time:.2f}s")

        self._cached_prompt_text = prompt_text

        if prompt_changed and self.offload_text_encoder:
            self._offload_text_encoder()

        # HeliosPipeline returns HeliosPipelineOutput with .frames attribute
        # .frames[0] is a list of PIL images or numpy arrays
        video = output.frames[0]

        video = self._convert_output_to_thwc(video)

        logger.info(f"Output video shape: {video.shape}")

        return {"video": video}

    def _convert_output_to_thwc(self, video) -> torch.Tensor:
        """Convert diffusers pipeline output to [T, H, W, C] float32 in [0, 1]."""
        import numpy as np

        if isinstance(video, list):
            # List of PIL images or numpy arrays
            frames = []
            for frame in video:
                if hasattr(frame, "convert"):
                    frame = np.array(frame.convert("RGB"))
                if isinstance(frame, np.ndarray):
                    frame = torch.from_numpy(frame)
                frames.append(frame)
            video = torch.stack(frames)
        elif isinstance(video, np.ndarray):
            video = torch.from_numpy(video)

        if not isinstance(video, torch.Tensor):
            video = torch.tensor(video)

        # Handle various output shapes
        if video.ndim == 4:
            if video.shape[-1] in (1, 3, 4):
                pass  # Already THWC
            elif video.shape[1] in (1, 3, 4):
                video = video.permute(0, 2, 3, 1)
        elif video.ndim == 5:
            video = video.squeeze(0)
            if video.shape[1] in (1, 3, 4) and video.shape[-1] not in (1, 3, 4):
                video = video.permute(0, 2, 3, 1)

        video = video.float()

        if video.max() > 1.5:
            video = video / 255.0

        return video.clamp(0.0, 1.0)
