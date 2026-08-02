"""Kohya-format LoRA merging for the SoulX-LiveAct DiT.

Why merge instead of running adapters as a side path
----------------------------------------------------
The streaming demo wraps ~480 of the DiT's nn.Linear modules in FP8Linear
(fp8_gemm.enable_fp8_gemm) and then torch.compile's every transformer block.
An adapter side path would add two bf16 matmuls per module that are neither
quantized nor inside the compiled graph, so it would change both the s/chunk
baseline and the graph shapes -- and then a LoRA A/B would really be measuring
the adapter's overhead. Merging leaves the FP8Linear count, every graph, and
the VRAM profile byte-identical to the no-LoRA run, so timings stay comparable.

Ordering is therefore load -> merge -> enable_fp8_gemm -> .to(cuda) -> compile.
Merging after the fp8 wrap would be writing bf16 deltas onto weights that are
already quantized; merging after compile invalidates the graphs.

Key naming
----------
Kohya/musubi checkpoints flatten the torch module path with underscores and
prefix it, e.g. `blocks.0.cross_attn.k` -> `lora_unet_blocks_0_cross_attn_k`.
That is NOT reversible by string surgery, because module names contain their
own underscores (`cross_attn`, `time_projection`, `self_attn`). So instead of
guessing, we walk the model's real nn.Linear modules, flatten each path the
same way, and look the LoRA key up in that table. Exact, and it makes an
architecture mismatch show up as an explicit unmatched-key report rather than
as a silently half-applied LoRA.

Verified against two checkpoints from NSFW-API (both `lora_unet_*`):
  nsfw_wan_14b_revealing_boobs  400 modules, rank 16,     bf16, blocks only
  nsfw_lora_wan_14b_e15         406 modules, rank 64/128, fp16, + time_*/head
SoulX's DiT is Wan2.1-14B-shaped (dim 5120, 40 blocks) with I2V cross-attention,
so `cross_attn.{q,k,v,o}` match while `k_img`/`v_img` (identity conditioning),
`audio_cross_attn` (lipsync) and `self_attn.memory_proj_*` (the streaming
memory) have no LoRA counterpart and are left untouched by construction.

Targets are tracked by module PATH, not by object reference: enable_fp8_gemm
does `setattr(parent, name, FP8Linear(...))`, which keeps the path but throws
away the original nn.Linear. Anything holding the old object would silently
write into a detached tensor.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_PREFIXES = ("lora_unet_", "diffusion_model.", "lora_")


def _strip_prefix(name: str) -> str:
    for p in _PREFIXES:
        if name.startswith(p):
            return name[len(p) :]
    return name


class LoRA:
    """A loaded kohya LoRA bound to one model, with a mutable strength.

    Holds the (down, up, scale) triplets on CPU and remembers the strength it
    has currently applied, so `set_strength` moves to a new value by applying
    only the *difference*. That is what makes a strength sweep cost one warmup
    instead of one per point, and it avoids keeping a ~28GB bf16 copy of the
    base weights around purely to be able to un-merge.
    """

    def __init__(self, path: str, model: nn.Module):
        self.path = path
        self.name = os.path.basename(path)
        self.strength = 0.0
        # (module path, down [r,in] fp32, up [out,r] fp32, alpha/rank)
        self._mods: List[Tuple[str, torch.Tensor, torch.Tensor, float]] = []
        self.unmatched: List[str] = []
        self.ranks: set = set()
        self._model = model
        self._load(model)

    @property
    def n_modules(self) -> int:
        return len(self._mods)

    # -- loading ---------------------------------------------------------

    def _load(self, model: nn.Module) -> None:
        from safetensors.torch import load_file

        sd = load_file(self.path, device="cpu")

        # flattened path -> real path, for every Linear in the model. Built
        # before the fp8 wrap, when the targets are still plain nn.Linear.
        table: Dict[str, str] = {
            name.replace(".", "_"): name
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear)
        }

        stems: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, tensor in sd.items():
            for suffix in ("lora_down.weight", "lora_up.weight", "alpha"):
                if key.endswith("." + suffix):
                    stem = _strip_prefix(key[: -len(suffix) - 1])
                    stems.setdefault(stem, {})[suffix] = tensor
                    break

        for stem, parts in sorted(stems.items()):
            down, up = parts.get("lora_down.weight"), parts.get("lora_up.weight")
            if down is None or up is None:
                self.unmatched.append(f"{stem} (incomplete pair)")
                continue
            path = table.get(stem)
            if path is None:
                self.unmatched.append(stem)
                continue
            weight = model.get_submodule(path).weight
            if up.shape[0] != weight.shape[0] or down.shape[1] != weight.shape[1]:
                self.unmatched.append(
                    f"{stem} (up{tuple(up.shape)} down{tuple(down.shape)} "
                    f"vs weight{tuple(weight.shape)})"
                )
                continue
            rank = down.shape[0]
            alpha = parts.get("alpha")
            alpha = float(alpha.item()) if alpha is not None else float(rank)
            self._mods.append((path, down.float(), up.float(), alpha / rank))
            self.ranks.add(rank)

    # -- merging ---------------------------------------------------------

    def _write(self, path: str, delta: torch.Tensor) -> None:
        """Add `delta` to the effective bf16 weight at `path`, wherever it lives.

        Before enable_fp8_gemm the target is a plain nn.Linear. Afterwards the
        bf16 master lives in FP8Linear._fp16_weight_cpu and the live weight is a
        cached fp8 quantization of it, so a strength change has to edit the
        master and force a requantize. FP8GemmOptions defaults to
        fp16_weight_storage="discard", which frees that master after the first
        forward -- interactive_demo selects "cpu_offload" when a LoRA is loaded
        so this path stays available.
        """
        mod = self._model.get_submodule(path)

        if isinstance(mod, nn.Linear):
            with torch.no_grad():
                w = mod.weight
                w.data += delta.to(dtype=w.dtype, device=w.device)
            return

        master = getattr(mod, "_fp16_weight_cpu", None)
        if master is None:
            if getattr(mod, "linear", None) is not None:  # "keep" mode
                with torch.no_grad():
                    w = mod.linear.weight
                    w.data += delta.to(dtype=w.dtype, device=w.device)
                mod.invalidate_weight_cache()
                return
            raise RuntimeError(
                f"{path}: FP8Linear has no bf16 master weight, so the LoRA "
                "strength cannot be changed at runtime (fp16_weight_storage="
                "'discard' frees it after the first forward). Relaunch with "
                "--lora so 'cpu_offload' is selected."
            )
        device = mod._cached_fp8_device()
        with torch.no_grad():
            master += delta.to(master.dtype)
        mod.invalidate_weight_cache()
        if device is not None:
            mod.materialize_fp8_weight(device)

    def set_strength(self, strength: float) -> float:
        """Move to `strength`, applying only the delta from the current value."""
        step = float(strength) - self.strength
        if abs(step) < 1e-9:
            return self.strength
        with torch.no_grad():
            for path, down, up, scale in self._mods:
                self._write(path, (up @ down) * (scale * step))
        self.strength = float(strength)
        return self.strength

    # -- reporting --------------------------------------------------------

    def summary(self) -> str:
        ranks = "/".join(str(r) for r in sorted(self.ranks))
        s = (
            f"lora: {self.name} -> {self.n_modules} modules matched, "
            f"rank {ranks}, strength {self.strength}"
        )
        if self.unmatched:
            s += f"\nlora: {len(self.unmatched)} UNMATCHED: " + ", ".join(
                self.unmatched[:4]
            )
            if len(self.unmatched) > 4:
                s += " ..."
        return s


def load_and_merge(
    model: nn.Module, path: str, strength: float, *, verbose: bool = True
) -> Optional[LoRA]:
    """Load `path` and merge it into `model` at `strength`. Call before fp8."""
    if not path:
        return None
    if not os.path.exists(path):
        raise SystemExit(f"--lora: no such file: {path}")
    lora = LoRA(path, model)
    if lora.n_modules == 0:
        raise SystemExit(
            f"--lora: {lora.name} matched 0 of its modules in this model -- "
            f"wrong architecture? unmatched sample: {lora.unmatched[:3]}"
        )
    lora.set_strength(strength)
    if verbose:
        print(lora.summary(), flush=True)
    return lora
