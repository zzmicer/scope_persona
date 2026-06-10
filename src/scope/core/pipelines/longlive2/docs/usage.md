# LongLive 2.0

LongLive 2.0 is a streaming autoregressive video diffusion model built on **Wan2.2-TI2V-5B**. It extends LongLive 1 (which was based on Wan2.1 1.3B) with a larger, higher-quality 5B backbone and NVFP4 quantized checkpoints for real-time generation on NVIDIA Blackwell GPUs.

## Hardware

The NVFP4 checkpoints require a **Blackwell GPU (RTX 5090 / RTX 50-series)**. NVFP4 is a 4-bit floating point format with hardware acceleration on Blackwell; it is not supported on older architectures.

- **`nvfp4-s2`** (2-step) — fastest, targets ~45fps. Recommended for interactive/real-time use.
- **`nvfp4-s4`** (4-step) — higher quality, slightly slower.
- **`bf16`** — portable fallback that runs on non-Blackwell GPUs but needs more VRAM and is slower. Use this if NVFP4 is unavailable.

Select the variant via the **Precision** setting.

## Modes

Wan2.2-TI2V-5B is a TI2V (text+image to video) model, so LongLive 2.0 supports **both**:

- **Text mode** — generate video from prompts alone.
- **Image mode** — supply a first frame; the model conditions generation on it (first-frame conditioning) and continues the motion from there.

## Resolution

The model was trained around 720p (e.g. 1280x704). Smaller resolutions generate faster (higher FPS, smoother streaming); larger resolutions improve visual quality at the cost of throughput. Dimensions must be multiples of 32 (Wan2.2 VAE 16x spatial downsample x 2x patch embedding).

## Seed

The seed parameter reproduces generations. If you like a result for a given seed and prompt sequence, reuse that seed with the same prompts to reproduce it.

## Prompting

The same prompting guidance as LongLive 1 applies (see the [original LongLive repo](https://github.com/NVlabs/LongLive)):

- Include a **subject** (who/what) and a **background/setting** (where) in each prompt; keep referencing the same subject/setting for continuity across shots.
- Prefer **cinematic long takes** over rapid shot-by-shot cuts.
- Use **long, detailed prompts** — a base prompt expanded by an LLM chatbot works well.

## Offline Generation

If the model weights are not downloaded yet:

```
# Run from scope directory
uv run download_models --pipeline longlive2
```
