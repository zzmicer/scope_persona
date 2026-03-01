# Weight loading utilities for Lynx IP-Adapter.
#
# Handles loading resampler.safetensors and ip_layers.safetensors
# from the ByteDance/lynx HuggingFace repo.

import logging

from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def _remap_feedforward_keys(state_dict):
    """Remap Lynx checkpoint FeedForward keys to match our Resampler structure.

    Lynx checkpoint uses plain Sequential indices: layers.X.1.0.weight
    Our FeedForward wraps in self.net: layers.X.1.net.0.weight
    """
    import re

    remapped = {}
    pattern = re.compile(r"^(layers\.\d+\.1)\.(\d+\..*)$")
    for key, value in state_dict.items():
        m = pattern.match(key)
        if m:
            remapped[f"{m.group(1)}.net.{m.group(2)}"] = value
        else:
            remapped[key] = value
    return remapped


def load_resampler_weights(resampler, resampler_path):
    """Load Lynx resampler weights from safetensors file.

    Args:
        resampler: Resampler module instance
        resampler_path: Path to resampler.safetensors
    """
    state_dict = load_file(resampler_path)
    state_dict = _remap_feedforward_keys(state_dict)
    resampler.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded resampler weights from {resampler_path}")


def load_ip_layer_weights(ip_cross_attn_layers, ip_layers_path):
    """Load Lynx IP cross-attention layer weights from safetensors file.

    The Lynx ip_layers.safetensors has keys like:
      {0}.to_k_ip.weight, {0}.to_v_ip.weight, {1}.to_k_ip.weight, ...

    These map directly to the nn.ModuleList indices.

    Args:
        ip_cross_attn_layers: nn.ModuleList of IPCrossAttentionLayer
        ip_layers_path: Path to ip_layers.safetensors
    """
    state_dict = load_file(ip_layers_path)
    ip_cross_attn_layers.load_state_dict(state_dict, strict=False)
    logger.info(
        f"Loaded IP cross-attention weights from {ip_layers_path} "
        f"({len(ip_cross_attn_layers)} layers)"
    )
