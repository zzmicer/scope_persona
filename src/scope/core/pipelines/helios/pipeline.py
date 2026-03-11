"""Helios Distilled autoregressive video generation pipeline for Daydream Scope.

Uses the HuggingFace diffusers HeliosPyramidPipeline to load and run the
Helios-Distilled model. Components (transformer, VAE, scheduler) are loaded
individually from the local models directory to avoid issues with
model_index.json auto-detection.

The model generates video autoregressively in 33-frame chunks via pyramid
multi-scale denoising with multi-term memory patchification. Each __call__
produces a single chunk (~33 pixel frames), and the pipeline processor calls
it repeatedly in a loop for continuous streaming.
"""

import dataclasses
import gc
import logging
import math
import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from scope.core.config import get_model_file_path
from scope.core.pipelines.interface import Pipeline

from .schema import HeliosConfig

if TYPE_CHECKING:
    from scope.core.pipelines.schema import BasePipelineConfig

logger = logging.getLogger(__name__)

MODEL_DIR_NAME = "Helios-Distilled"

# Helios autoregressive defaults
DEFAULT_HISTORY_SIZES = [16, 2, 1]
DEFAULT_NUM_LATENT_FRAMES_PER_CHUNK = 9


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


@dataclasses.dataclass
class _HeliosARState:
    """Autoregressive state maintained between chunk calls."""

    prompt_text: str
    prompt_embeds: torch.Tensor
    history_latents: torch.Tensor
    image_latents: torch.Tensor | None
    total_generated_latent_frames: int
    chunk_index: int

    # Generation config (fixed for the session)
    height: int
    width: int
    num_latent_frames_per_chunk: int
    history_sizes: list[int]
    num_history_latent_frames: int
    keep_first_frame: bool
    pyramid_num_inference_steps_list: list[int]
    is_amplify_first_chunk: bool
    generator: torch.Generator

    # Precomputed constants
    latents_mean: torch.Tensor
    latents_std: torch.Tensor
    num_channels_latents: int
    window_num_frames: int

    # Index tensors
    indices_hidden_states: torch.Tensor
    indices_latents_history_short: torch.Tensor
    indices_latents_history_mid: torch.Tensor
    indices_latents_history_long: torch.Tensor


class HeliosPipeline(Pipeline):
    """Helios Distilled video generation pipeline.

    Wraps the diffusers HeliosPyramidPipeline for use within Daydream Scope.
    The model generates video autoregressively — each __call__ produces one
    chunk of 33 pixel frames (9 latent frames) using pyramid multi-scale
    denoising with only 2 steps per stage.

    The pipeline processor calls __call__ repeatedly in a loop. State
    (history latents, image anchor) is maintained between calls for temporal
    coherence across chunks.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return HeliosConfig

    def __init__(
        self,
        height: int = 384,
        width: int = 640,
        pyramid_steps: int = 2,
        amplify_first_chunk: bool = True,
        offload_text_encoder: bool = True,
        compile: bool = True,
        attention_backend: str | None = None,
        compile_vae: bool = False,
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
        self.pyramid_steps = pyramid_steps
        self.amplify_first_chunk = amplify_first_chunk
        self.offload_text_encoder = offload_text_encoder
        self.compile = compile
        self.attention_backend = attention_backend
        self.compile_vae = compile_vae

        if kwargs:
            logger.debug(
                f"HeliosPipeline ignoring unknown kwargs: {list(kwargs.keys())}"
            )

        start = time.time()
        model_path = str(get_model_file_path(MODEL_DIR_NAME))
        logger.info(f"Loading Helios Distilled from: {model_path}")

        self._pipe = self._load_pipeline(model_path)

        logger.info(f"Helios Distilled loaded in {time.time() - start:.2f}s")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory after load: {allocated:.2f} GB")

        # Autoregressive state (initialized on first generate call)
        self._ar_state: _HeliosARState | None = None

    def _load_pipeline(self, model_path: str):
        """Load the Helios diffusers pipeline with explicit component loading."""
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

        # --- Model optimizations ---

        # 1. Fuse QKV projections (~15-25% speedup)
        try:
            pipe.transformer.fuse_qkv_projections()
            logger.info("  Fused QKV projections")
        except Exception as e:
            logger.warning(f"  Failed to fuse QKV projections: {e}")

        # 2. Set attention backend (FA3 for H100+, FA2 for Ampere, default SDPA)
        attention_backend = self.attention_backend
        if attention_backend is None and torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc[0] >= 9:  # H100+ (sm_90)
                attention_backend = "_FLASH_3_HUB"
            else:
                attention_backend = "FLASH_HUB"  # FA2 for Ampere

        if attention_backend is not None:
            try:
                pipe.transformer.set_attention_backend(attention_backend)
                logger.info(f"  Attention backend set to: {attention_backend}")
            except Exception as e:
                logger.warning(
                    f"  Failed to set attention backend '{attention_backend}': {e}"
                )
                # Try fallback chain: FA3 -> FA2 -> leave default
                if attention_backend == "_FLASH_3_HUB":
                    try:
                        pipe.transformer.set_attention_backend("FLASH_HUB")
                        logger.info("  Fell back to FLASH_HUB (FA2)")
                    except Exception as e2:
                        logger.warning(f"  FA2 fallback also failed: {e2}")

        # 3. Compile transformer blocks (regional compilation, ~1.3-1.8x speedup)
        if self.compile:
            try:
                pipe.transformer.compile_repeated_blocks(
                    mode="max-autotune-no-cudagraphs", fullgraph=False
                )
                logger.info("  Compiled transformer repeated blocks")
            except Exception as e:
                logger.warning(f"  Failed to compile transformer blocks: {e}")

        # 4. Compile VAE decoder (optional, lower priority)
        if self.compile_vae:
            try:
                pipe.vae.decode = torch.compile(
                    pipe.vae.decode, mode="reduce-overhead", fullgraph=False
                )
                logger.info("  Compiled VAE decoder")
            except Exception as e:
                logger.warning(f"  Failed to compile VAE decoder: {e}")

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
            if (
                hasattr(self._pipe, "text_encoder")
                and self._pipe.text_encoder is not None
            ):
                if next(self._pipe.text_encoder.parameters()).device.type == "cpu":
                    self._pipe.text_encoder.to(self.device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    logger.info("Text encoder moved back to GPU")
        except Exception as e:
            logger.warning(f"Failed to move text encoder to device: {e}")

    def __call__(self, **kwargs) -> dict:
        """Generate one chunk of video from text prompt.

        Each call produces ~33 pixel frames (one autoregressive chunk).
        State is maintained between calls for temporal coherence.

        Args:
            **kwargs: Generation parameters including:
                - prompts: List of prompt dicts with 'text' and 'weight' keys
                - base_seed: Random seed for generation
                - height/width: Override resolution
                - pyramid_steps: Override denoising steps per stage
                - amplify_first_chunk: Override first chunk amplification
                - init_cache: If True, reinitialize autoregressive state

        Returns:
            Dictionary with "video" key containing [T, H, W, C] tensor in [0, 1] range.
        """
        return self._generate(**kwargs)

    @torch.inference_mode()
    def _generate(self, **kwargs) -> dict:
        prompts = kwargs.get(
            "prompts", [{"text": "a beautiful landscape", "weight": 1.0}]
        )
        prompt_text = prompts[0]["text"] if prompts else "a beautiful landscape"
        seed = kwargs.get("base_seed", kwargs.get("seed", 42))
        height = kwargs.get("height", self.height)
        width = kwargs.get("width", self.width)
        pyramid_steps = kwargs.get("pyramid_steps", self.pyramid_steps)
        amplify_first_chunk = kwargs.get(
            "amplify_first_chunk", self.amplify_first_chunk
        )
        init_cache = kwargs.get("init_cache", False)

        # Determine if we need to (re)initialize autoregressive state
        needs_init = (
            init_cache
            or self._ar_state is None
            or prompt_text != self._ar_state.prompt_text
            or height != self._ar_state.height
            or width != self._ar_state.width
        )

        if needs_init:
            self._setup_autoregressive(
                prompt_text=prompt_text,
                seed=seed,
                height=height,
                width=width,
                pyramid_steps=pyramid_steps,
                amplify_first_chunk=amplify_first_chunk,
            )

        return self._generate_chunk()

    def _setup_autoregressive(
        self,
        prompt_text: str,
        seed: int,
        height: int,
        width: int,
        pyramid_steps: int,
        amplify_first_chunk: bool,
    ) -> None:
        """Initialize autoregressive state for a new generation session."""
        logger.info(
            f"Setting up autoregressive generation at {height}x{width}, seed={seed}"
        )

        pipe = self._pipe
        device = self.device

        # Text encoding
        self._ensure_text_encoder_on_device()

        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt_text,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(pipe.transformer.dtype)

        if self.offload_text_encoder:
            self._offload_text_encoder()

        # VAE constants
        vae_dtype = pipe.vae.dtype
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(device, vae_dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(device, vae_dtype)

        # Autoregressive config
        num_latent_frames_per_chunk = DEFAULT_NUM_LATENT_FRAMES_PER_CHUNK
        history_sizes = sorted(DEFAULT_HISTORY_SIZES, reverse=True)
        keep_first_frame = True
        num_channels_latents = pipe.transformer.config.in_channels
        window_num_frames = (
            num_latent_frames_per_chunk - 1
        ) * pipe.vae_scale_factor_temporal + 1
        num_history_latent_frames = sum(history_sizes)

        # Initialize history latents (zeros)
        history_latents = torch.zeros(
            1,
            num_channels_latents,
            num_history_latent_frames,
            height // pipe.vae_scale_factor_spatial,
            width // pipe.vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )

        # Prepare index tensors for history conditioning
        if keep_first_frame:
            indices = torch.arange(
                0, sum([1, *history_sizes, num_latent_frames_per_chunk])
            )
            (
                indices_prefix,
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_1x,
                indices_hidden_states,
            ) = indices.split([1, *history_sizes, num_latent_frames_per_chunk], dim=0)
            indices_latents_history_short = torch.cat(
                [indices_prefix, indices_latents_history_1x], dim=0
            )
        else:
            indices = torch.arange(
                0, sum([*history_sizes, num_latent_frames_per_chunk])
            )
            (
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_short,
                indices_hidden_states,
            ) = indices.split([*history_sizes, num_latent_frames_per_chunk], dim=0)

        steps_list = [pyramid_steps] * 3

        generator = torch.Generator(device).manual_seed(seed)

        self._ar_state = _HeliosARState(
            prompt_text=prompt_text,
            prompt_embeds=prompt_embeds,
            history_latents=history_latents,
            image_latents=None,
            total_generated_latent_frames=0,
            chunk_index=0,
            height=height,
            width=width,
            num_latent_frames_per_chunk=num_latent_frames_per_chunk,
            history_sizes=history_sizes,
            num_history_latent_frames=num_history_latent_frames,
            keep_first_frame=keep_first_frame,
            pyramid_num_inference_steps_list=steps_list,
            is_amplify_first_chunk=amplify_first_chunk,
            generator=generator,
            latents_mean=latents_mean,
            latents_std=latents_std,
            num_channels_latents=num_channels_latents,
            window_num_frames=window_num_frames,
            indices_hidden_states=indices_hidden_states.unsqueeze(0),
            indices_latents_history_short=indices_latents_history_short.unsqueeze(0),
            indices_latents_history_mid=indices_latents_history_mid.unsqueeze(0),
            indices_latents_history_long=indices_latents_history_long.unsqueeze(0),
        )

        logger.info(
            f"Autoregressive state initialized: "
            f"chunk_frames={window_num_frames}, "
            f"pyramid_steps={steps_list}"
        )

    def _generate_chunk(self) -> dict:
        """Generate a single autoregressive chunk and update state.

        Extracts one iteration of the diffusers HeliosPyramidPipeline's
        internal autoregressive loop: prepare history conditioning, denoise
        via pyramid multi-scale stages, VAE-decode, and update history.

        Returns:
            Dictionary with "video" key containing [T, H, W, C] float32 tensor in [0, 1].
        """
        state = self._ar_state
        pipe = self._pipe
        device = self.device
        transformer_dtype = pipe.transformer.dtype
        vae_dtype = pipe.vae.dtype

        k = state.chunk_index
        is_first_chunk = k == 0

        chunk_start = time.perf_counter()

        # --- Extract history conditioning ---
        if state.keep_first_frame:
            latents_history_long, latents_history_mid, latents_history_1x = (
                state.history_latents[:, :, -state.num_history_latent_frames :].split(
                    state.history_sizes, dim=2
                )
            )
            if state.image_latents is None and is_first_chunk:
                latents_prefix = torch.zeros(
                    (
                        1,
                        state.num_channels_latents,
                        1,
                        latents_history_1x.shape[-2],
                        latents_history_1x.shape[-1],
                    ),
                    device=device,
                    dtype=latents_history_1x.dtype,
                )
            else:
                latents_prefix = state.image_latents
            latents_history_short = torch.cat(
                [latents_prefix, latents_history_1x], dim=2
            )
        else:
            latents_history_long, latents_history_mid, latents_history_short = (
                state.history_latents[:, :, -state.num_history_latent_frames :].split(
                    state.history_sizes, dim=2
                )
            )

        t_history = time.perf_counter()

        # --- Prepare noise latents for this chunk ---
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=state.num_channels_latents,
            height=state.height,
            width=state.width,
            num_frames=state.window_num_frames,
            dtype=torch.float32,
            device=device,
            generator=state.generator,
        )
        t_noise = time.perf_counter()

        # --- Pyramid multi-scale denoising ---
        pyramid_num_stages = len(state.pyramid_num_inference_steps_list)
        pyramid_stage_times = []
        num_latent_frames_per_chunk = state.num_latent_frames_per_chunk

        _, _, _, pyramid_height, pyramid_width = latents.shape

        # Downscale latents to lowest pyramid resolution
        latents = latents.permute(0, 2, 1, 3, 4).reshape(
            num_latent_frames_per_chunk,
            state.num_channels_latents,
            pyramid_height,
            pyramid_width,
        )
        for _ in range(pyramid_num_stages - 1):
            pyramid_height //= 2
            pyramid_width //= 2
            latents = (
                F.interpolate(
                    latents,
                    size=(pyramid_height, pyramid_width),
                    mode="bilinear",
                )
                * 2
            )
        latents = latents.reshape(
            1,
            num_latent_frames_per_chunk,
            state.num_channels_latents,
            pyramid_height,
            pyramid_width,
        ).permute(0, 2, 1, 3, 4)

        start_point_list = None
        if pipe.config.is_distilled:
            start_point_list = [latents]

        for stage_idx in range(pyramid_num_stages):
            stage_start = time.perf_counter()
            patch_size = pipe.transformer.config.patch_size
            image_seq_len = (
                latents.shape[-1] * latents.shape[-2] * latents.shape[-3]
            ) // (patch_size[0] * patch_size[1] * patch_size[2])

            mu = _calculate_shift(
                image_seq_len,
                pipe.scheduler.config.get("base_image_seq_len", 256),
                pipe.scheduler.config.get("max_image_seq_len", 4096),
                pipe.scheduler.config.get("base_shift", 0.5),
                pipe.scheduler.config.get("max_shift", 1.15),
            )
            pipe.scheduler.set_timesteps(
                state.pyramid_num_inference_steps_list[stage_idx],
                stage_idx,
                device=device,
                mu=mu,
                is_amplify_first_chunk=state.is_amplify_first_chunk and is_first_chunk,
            )
            timesteps = pipe.scheduler.timesteps

            if stage_idx > 0:
                pyramid_height *= 2
                pyramid_width *= 2
                latents = latents.permute(0, 2, 1, 3, 4).reshape(
                    num_latent_frames_per_chunk,
                    state.num_channels_latents,
                    pyramid_height // 2,
                    pyramid_width // 2,
                )
                latents = F.interpolate(
                    latents,
                    size=(pyramid_height, pyramid_width),
                    mode="nearest",
                )
                latents = latents.reshape(
                    1,
                    num_latent_frames_per_chunk,
                    state.num_channels_latents,
                    pyramid_height,
                    pyramid_width,
                ).permute(0, 2, 1, 3, 4)

                # Fix block artifacts at stage transition
                ori_sigma = 1 - pipe.scheduler.ori_start_sigmas[stage_idx]
                gamma = pipe.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                noise = pipe.sample_block_noise(
                    1,
                    state.num_channels_latents,
                    num_latent_frames_per_chunk,
                    pyramid_height,
                    pyramid_width,
                    patch_size,
                    device,
                )
                noise = noise.to(device=device, dtype=transformer_dtype)
                latents = alpha * latents + beta * noise

                if pipe.config.is_distilled:
                    start_point_list.append(latents)

            # Denoising steps within this pyramid stage
            for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(torch.int64)

                latent_model_input = latents.to(transformer_dtype)
                hist_short = latents_history_short.to(transformer_dtype)
                hist_mid = latents_history_mid.to(transformer_dtype)
                hist_long = latents_history_long.to(transformer_dtype)

                with pipe.transformer.cache_context("cond"):
                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=state.prompt_embeds,
                        indices_hidden_states=state.indices_hidden_states,
                        indices_latents_history_short=state.indices_latents_history_short,
                        indices_latents_history_mid=state.indices_latents_history_mid,
                        indices_latents_history_long=state.indices_latents_history_long,
                        latents_history_short=hist_short,
                        latents_history_mid=hist_mid,
                        latents_history_long=hist_long,
                        return_dict=False,
                    )[0]

                # Helios Distilled: guidance_scale=1.0, no CFG needed

                extra_kwargs = (
                    {
                        "cur_sampling_step": i,
                        "dmd_noisy_tensor": start_point_list[stage_idx]
                        if start_point_list is not None
                        else None,
                        "dmd_sigmas": pipe.scheduler.sigmas,
                        "dmd_timesteps": pipe.scheduler.timesteps,
                        "all_timesteps": timesteps,
                    }
                    if pipe.config.is_distilled
                    else {}
                )

                latents = pipe.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    generator=state.generator,
                    return_dict=False,
                    **extra_kwargs,
                )[0]

            pyramid_stage_times.append(time.perf_counter() - stage_start)

        t_pyramid = time.perf_counter()
        # --- Capture first-frame anchor ---
        if state.keep_first_frame and is_first_chunk and state.image_latents is None:
            state.image_latents = latents[:, :, 0:1, :, :]

        # --- Update history ---
        state.total_generated_latent_frames += latents.shape[2]
        state.history_latents = torch.cat([state.history_latents, latents], dim=2)

        # --- VAE decode this chunk ---
        real_history_latents = state.history_latents[
            :, :, -state.total_generated_latent_frames :
        ]
        current_latents = (
            real_history_latents[:, :, -num_latent_frames_per_chunk:].to(vae_dtype)
            / state.latents_std
            + state.latents_mean
        )
        current_video = pipe.vae.decode(current_latents, return_dict=False)[0]

        # Trim to valid temporal alignment
        generated_frames = current_video.size(2)
        generated_frames = (
            generated_frames - 1
        ) // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
        current_video = current_video[:, :, :generated_frames]
        t_vae = time.perf_counter()

        # Convert from BCTHW [-1,1] to THWC [0,1]
        video = pipe.video_processor.postprocess_video(current_video, output_type="np")[
            0
        ]
        video = self._convert_output_to_thwc(video)
        t_post = time.perf_counter()

        chunk_time = t_post - chunk_start
        stage_str = " | ".join(f"s{i}={pyramid_stage_times[i]:.3f}s" for i in range(len(pyramid_stage_times)))
        logger.info(
            f"[helios] chunk={k} total={chunk_time:.3f}s "
            f"hist={t_history - chunk_start:.3f}s "
            f"noise={t_noise - t_history:.3f}s "
            f"pyramid=[{stage_str}] "
            f"vae={t_vae - t_pyramid:.3f}s "
            f"post={t_post - t_vae:.3f}s "
            f"frames={video.shape[0]}"
        )

        state.chunk_index += 1

        return {"video": video}

    def _convert_output_to_thwc(self, video) -> torch.Tensor:
        """Convert diffusers pipeline output to [T, H, W, C] float32 in [0, 1]."""
        import numpy as np

        if isinstance(video, list):
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
