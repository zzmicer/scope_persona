"""Wan2.2-TI2V-5B component layer.

Mirrors ``scope.core.pipelines.wan2_1`` but targets the Wan2.2-TI2V-5B base
model used by LongLive 2.0. This is a Mac scaffold: structure is ported but the
heavy transformer / VAE numerics are STUBS to be filled in on a CUDA box.

See ``README.md`` in this directory for the exact ported-vs-stubbed breakdown
and upstream file mapping (NVlabs/LongLive ``wan_5b/``).
"""
