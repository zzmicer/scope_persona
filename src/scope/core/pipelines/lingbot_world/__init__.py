"""LingBot-World-V2 interactive world pipeline.

Wraps https://github.com/Robbyant/lingbot-world-v2 (14B causal-fast,
Wan2.2-based) for interactive, text-controlled world generation from a
single image: camera motion is driven by synthesized pose trajectories
(Plücker embeddings) and scene events are driven by prompt swaps over a
persistent KV cache.

The heavy modules (`session`, `demo`) import the upstream `wan` package
from a lingbot-world-v2 checkout at runtime; nothing here is imported at
scope server startup.
"""
