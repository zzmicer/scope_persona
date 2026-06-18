# Adapted from ../wan2_1/components/generator.py for the Wan2.2-TI2V-5B base model.
# Upstream reference: NVlabs/LongLive `wan_5b/modules/causal_model.py` + `model.py`.
#
# This loader is mechanically ported from wan2_1. The flow-matching math
# (_convert_flow_pred_to_x0 / scheduler) is model-agnostic and carried over
# unchanged. The places that genuinely differ for the 5B model are the config
# defaults and the RoPE freq-table reconstruction, which depend on the 5B
# head_dim. Those are marked with `# TODO(longlive2):` and cite the exact
# upstream symbol to port.
import inspect
import json
import os
import types

import torch

from scope.core.pipelines.utils import load_state_dict

# NOTE: wan2_2 ships its own scheduler-free path for now; we reuse the wan2_1
# FlowMatchScheduler since the flow-matching formulation is identical between
# Wan2.1 and Wan2.2 (both are rectified-flow / flow-matching models).
# TODO(longlive2): if a wan2_2-specific scheduler is needed, port it alongside
# this file; otherwise this import is intentional reuse, not an oversight.
from ...wan2_1.components.scheduler import FlowMatchScheduler, SchedulerInterface


def filter_causal_model_cls_config(causal_model_cls, config):
    # Filter config to only include parameters accepted by the model's __init__
    sig = inspect.signature(causal_model_cls.__init__)
    config = {k: v for k, v in config.items() if k in sig.parameters}
    return config


class WanDiffusionWrapper(torch.nn.Module):
    """WanDiffusionWrapper for the Wan2.2-TI2V-5B base model.

    Differences vs the wan2_1 wrapper (see upstream
    `wan_5b/configs/wan_ti2v_5B.py` and `wan_5b/modules/causal_model.py`):

      - 5B transformer dims: dim=3072, ffn_dim=14336, num_heads=24,
        num_layers=30, freq_dim=256, patch_size=(1, 2, 2), eps=1e-6,
        qk_norm=True, cross_attn_norm=True.
      - in_dim == out_dim == 48 (the Wan2.2 VAE latent channel count), vs 16
        for Wan2.1. This propagates into patch_embedding and Head shapes.
      - VAE stride is (4, 16, 16) -> 16x spatial / 4x temporal compression,
        so the per-frame token grid (and therefore seq_len / freqs sizing)
        differs from Wan2.1's 8x spatial VAE.

    The real numeric constants are read from the model's `config.json` at load
    time wherever possible; we do NOT hardcode tensor shapes we cannot verify
    from upstream.
    """

    def __init__(
        self,
        causal_model_cls,
        model_name="Wan2.2-TI2V-5B",
        timestep_shift=5.0,  # upstream sample_shift for 5B (wan_ti2v_5B.py)
        local_attn_size=-1,
        sink_size=0,
        model_dir: str | None = None,
        generator_path: str | None = None,
        generator_model_name: str | None = None,
        **model_kwargs,
    ):
        super().__init__()

        # Use provided model_dir or default to "wan_models"
        model_dir = model_dir if model_dir is not None else "wan_models"
        model_path = os.path.join(model_dir, f"{model_name}/")

        if generator_path:
            config_path = os.path.join(model_path, "config.json")
            with open(config_path) as f:
                config = json.load(f)

            config.update({"local_attn_size": local_attn_size, "sink_size": sink_size})
            # Merge in additional model-specific kwargs.
            config.update(model_kwargs)

            state_dict = load_state_dict(generator_path)
            # Handle case where the required keys are nested, e.g. state_dict["generator"]
            if generator_model_name is not None:
                state_dict = state_dict[generator_model_name]

            # Remove 'model.' prefix if present (from wrapped models)
            if all(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("model.", "", 1): v for k, v in state_dict.items()
                }

            with torch.device("meta"):
                self.model = causal_model_cls(
                    **filter_causal_model_cls_config(causal_model_cls, config)
                )

            # HACK! Store freqs shape before to_empty() wipes the buffer.
            freqs_shape = (
                self.model.freqs.shape if hasattr(self.model, "freqs") else None
            )

            # Materialize buffers/params on CPU, then assign weights.
            self.model = self.model.to_empty(device="cpu")
            self.model.load_state_dict(state_dict, assign=True, strict=False)

            # HACK! Reinitialize self.freqs on CPU (not stored in state_dict).
            if freqs_shape is not None and hasattr(self.model, "freqs"):
                d = self.model.dim // self.model.num_heads

                # RoPE split VERIFIED against wan_5b/modules/model.py
                # (CausalWanModel.__init__): the freqs table is
                #   cat([rope_params(1024, d - 4*(d//6)),
                #        rope_params(1024, 2*(d//6)),
                #        rope_params(1024, 2*(d//6))], dim=1)
                # with theta=10000. The split formula is parametric in head_dim,
                # so it is identical to the 1.3B split. For 5B head_dim=128:
                # d//6=21 -> temporal=44, spatial_h=42, spatial_w=42 (== 128).
                # This reconstruction matches CausalWanModel.freqs in
                # ../modules/causal_model.py exactly.
                def rope_params(max_seq_len, dim, theta=10000):
                    assert dim % 2 == 0
                    freqs = torch.outer(
                        torch.arange(max_seq_len),
                        1.0
                        / torch.pow(
                            theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)
                        ),
                    )
                    freqs = torch.polar(torch.ones_like(freqs), freqs)
                    return freqs

                self.model.freqs = torch.cat(
                    [
                        rope_params(1024, d - 4 * (d // 6)),
                        rope_params(1024, 2 * (d // 6)),
                        rope_params(1024, 2 * (d // 6)),
                    ],
                    dim=1,
                )
        else:
            from_pretrained_config = {
                "local_attn_size": local_attn_size,
                "sink_size": sink_size,
            }
            from_pretrained_config.update(model_kwargs)
            self.model = causal_model_cls.from_pretrained(
                model_path,
                **filter_causal_model_cls_config(
                    causal_model_cls,
                    from_pretrained_config,
                ),
            )

        self.model.eval()
        self.model.requires_grad_(False)

        # For non-causal diffusion, all frames share the same timestep.
        self.uniform_timestep = False

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # `seq_len` is ONLY consumed as the upper-bound assertion
        # `assert seq_lens.max() <= seq_len` inside the model forward; the model
        # never uses it to size any tensor (RoPE/KV-cache size from
        # local_attn_size * runtime frame_seqlen). So this just needs to be a
        # generous bound that never falsely trips.
        #
        # 5B geometry (wan_ti2v_5B.py): VAE stride (4, 16, 16) + patch_size
        # (1, 2, 2) => spatial token grid is H/32 x W/32, i.e. frame_seqlen =
        # (H/32) * (W/32). At a generous 1920x1920 ceiling that is 60*60 = 3600
        # tokens/frame; we use that per-frame ceiling so any practical
        # resolution fits. With local attention, a streaming forward only ever
        # sees up to `local_attn_size` frames at once; with global attention we
        # fall back to the wan2_1 global cap.
        _MAX_FRAME_SEQLEN_5B = 3600  # (H/32)*(W/32) ceiling for <= 1920x1920
        self.seq_len = (
            _MAX_FRAME_SEQLEN_5B * local_attn_size
            if local_attn_size != -1
            else _MAX_FRAME_SEQLEN_5B * 121  # full 5B clip (frame_num=121) ceiling
        )
        self.post_init()

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Convert flow-matching prediction to x0 prediction.

        Identical to Wan2.1: x0 = x_t - sigma_t * pred. Flow-matching
        formulation is shared between Wan2.1 and Wan2.2.
        """
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = [
            x.double().to(flow_pred.device)
            for x in [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        ]
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Convert x0 prediction to flow-matching prediction (shared with Wan2.1)."""
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = [
            x.double().to(x0_pred.device)
            for x in [x0_pred, xt, scheduler.sigmas, scheduler.timesteps]
        ]
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def _call_model(self, *args, **kwargs):
        # Inspect _forward_inference to filter kwargs the underlying model accepts.
        sig = inspect.signature(self.model._forward_inference)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_keyword:
            accepted = kwargs
        else:
            accepted = {
                name: value for name, value in kwargs.items() if name in sig.parameters
            }
        return self.model(*args, **accepted)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int | None = None,
        current_end: int | None = None,
        cache_start: int | None = None,
        **extra_kwargs,
    ) -> torch.Tensor:
        """Run one denoising step.

        Trimmed relative to wan2_1: the 5B upstream `_forward_inference`
        (wan_5b/modules/causal_model.py) does NOT expose the wan2_1
        kv_bank / GAN-classify / VACE arguments. `_call_model` already filters
        kwargs by signature, so any extra_kwargs that the 5B model doesn't
        accept are dropped safely.

        TODO(longlive2): confirm the exact `_forward_inference` arg set for 5B
        (x, t, context, seq_len, kv_cache, crossattn_cache, current_start,
        cache_start, defer_cache_updates) and wire `defer_cache_updates`
        through if the LongLive 2.0 pipeline needs it.
        """
        prompt_embeds = conditional_dict["prompt_embeds"]

        input_timestep = timestep[:, 0] if self.uniform_timestep else timestep

        if kv_cache is not None:
            flow_pred = self._call_model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                cache_start=cache_start,
                **extra_kwargs,
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self._call_model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                **extra_kwargs,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """Bind the SchedulerInterface conversion helpers onto self.scheduler."""
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler
        )
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """Post-construction hook: bind scheduler helper methods."""
        self.get_scheduler()
