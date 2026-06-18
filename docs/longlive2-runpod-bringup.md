# LongLive 2.0 — RunPod Blackwell bring-up (session findings)

**Date:** 2026-06-17
**Branch:** `longlive2`
**Goal:** Run the new `longlive2` (Wan2.2-TI2V-5B) pipeline on a real Blackwell GPU
and fix the "nothing renders / ICE failed" WebRTC bug.

## TL;DR status

- ✅ **WebRTC / TURN bug fixed and validated.** Behind RunPod's firewall STUN is
  insufficient; the server now feeds **Cloudflare TURN relays** into the ICE
  config (`RTCConfiguration created with cloudflare (direct keys)`). This is the
  `webrtc.py` fix committed earlier (`10e9014`).
- ✅ **`longlive2` renders correctly in BF16** on RTX PRO 6000 Blackwell (sm_120)
  after fixing several scaffold bugs (below). A coherent 3D-animated park scene is
  produced from the default panda prompt.
- ⛔ **NVFP4 path not yet working** (the 45fps Blackwell target). Blocked on two
  native-build issues (below). BF16 is the working fallback for now.
- ⚠️ **Published image `zmiccer/scope_persona:latest` ships WITHOUT a CUDA
  compiler/headers** despite the Dockerfile's `-devel-` base — must be fixed in CI.

## Bugs found & fixed (committed on `longlive2`)

All four are real scaffold bugs exposed by the first-ever run on GPU hardware.

1. **`model.yaml` never loaded → `max_rope_freq_table_seq_len` KeyError.**
   `load_model_config()` does `getattr(config, "model_config")` first, but
   `BasePipelineConfig` is a Pydantic v2 model whose **reserved** `model_config =
   ConfigDict(extra="forbid")` is truthy. So the helper returned Pydantic's
   ConfigDict instead of `model.yaml`, and `components.config` lacked all model
   params. LongLive 1 dodged this only because it passes an **OmegaConf** config.
   **Fix:** `longlive2/pipeline.py` now loads `model.yaml` explicitly via
   `OmegaConf.load(Path(__file__).parent / "model.yaml")`.

2. **Latent channels hardcoded to 16 (Wan2.1) — model expects 48 (Wan2.2).**
   `Conv3d patch_embedding expected 48 channels, got 16`.
   **Fix:** `wan2_1/blocks/prepare_latents.py` and
   `longlive/blocks/recache_frames.py` now read
   `getattr(components.config, "latent_channels", 16)`; `longlive2/model.yaml`
   declares `latent_channels: 48`. Default 16 keeps LongLive 1 unaffected.

3. **(same root as #2)** recache buffer mismatch in `prepare_recache_frames.py`
   `torch.cat` (16 vs 48) — fixed by the `recache_frames.py` allocation change.

4. **Noise output — wrong denoising schedule.** BF16 (and nvfp4-s4) are **4-step**
   DMD-distilled; the frontend sends a generic 2-step list `[700, 500]` (tuned for
   the 1.3B LongLive 1), which produces **pure noise** on the 5B model. The 4-step
   schedule `[1000, 750, 500, 250]` (from `model.yaml`) renders correctly.
   **Fix:** `longlive2/pipeline.py` precomputes the precision-correct schedule
   (`self._denoising_step_list`, subsampling `model.yaml denoising_steps` to
   `resolve_steps()`) and `_generate` **overrides** any incoming list whose length
   doesn't match. Verified: input `[700,500]` → overridden → coherent image.

   > UPDATE (2026-06-18): the real 5B 4-step schedule is now **confirmed from
   > upstream** `NVlabs/LongLive` (`configs/nvfp4/train_dmd_nvfp4_step4.yaml`):
   > `denoising_step_list: [1000, 946, 854, 681, 0]` with `warp_denoising_step:
   > false`. That is the "boundaries" convention (5 entries for `sampling_steps:4`);
   > our denoise loop uses the equivalent last-entry-is-final convention, so it maps
   > to `denoising_steps: [1000, 946, 854, 681]` (trailing 0 dropped). `model.yaml`
   > is now updated to this front-loaded schedule (was the wrong evenly-spaced 1.3B
   > `[1000,750,500,250]`). The **2-step** nvfp4-s2 schedule is still unverified —
   > upstream ships only `sampling_steps:4` configs; the s2 subsample now gives
   > `[1000, 681]` (real values, heuristic).

## NVFP4 blockers (for the 45fps target)

The default precision is `nvfp4-s2`. To run it on Blackwell we need 3 native
components built. Current state on the pod:

- ✅ `kv_dequant` CUDA kernel — **builds** (sm_120a) once CUDA dev headers present.
- ✅ `fouroversix` — installs, **but fails to import**:
  `ImportError: cannot import name 'WeightConverter' from 'transformers'`
  → a `transformers` version conflict. **RESOLVED (2026-06-18, pending pod
  validation):** `WeightConverter` was added in **transformers 5.0.0**; fouroversix
  is now integrated into transformers v5 (`FourOverSixConfig`) and requires it. Our
  base lock pins `transformers 4.57.5` (kept, to not disturb the working BF16 path).
  Scope's only live transformers usage is `AutoTokenizer` (v5-safe) and nothing in
  the lock caps transformers `<5` (only `daydream-scope >=4.49` and `peft` with no
  cap). Fix: `entrypoint.sh` now upgrades `transformers>=5.0.0` **in the NVFP4-only
  branch** (runs before the server starts, so the umt5 tokenizer also runs on v5)
  immediately before installing fouroversix. Still needs a render-validation pass on
  the pod to confirm the umt5 text encoder + wan pipelines tolerate v5 at runtime.
- ⛔ `transformer-engine[pytorch]` — builds from source (no prebuilt wheel for
  cu12/torch2.9/cp312) and needs CUDA math + cudnn dev headers. After installing
  `cuda-libraries-dev-12-8` it still needed `cudnn.h` (`libcudnn9-dev-cuda-12`,
  unhold required). Build was in progress when we pivoted to BF16.

When NVFP4 is unavailable, `longlive2/pipeline.py` already falls back to a bf16
cast — but note that fallback uses the **nvfp4 checkpoint** (`model_te.pt`) cast to
bf16, which is NOT the same as the real `model_bf16.pt`. For correct BF16 use
`precision="bf16"` explicitly.

## Image defect (fix in CI / Dockerfile)

`zmiccer/scope_persona:latest` had **no `nvcc` and no CUDA headers**
(`/usr/local/cuda-12.8` had no `bin/` or `include/`) even though the Dockerfile
runtime stage is `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04`. The published image
behaved like a `runtime` base — likely `:latest` predates the devel base or the
toolkit was stripped.

**FIXED (2026-06-18) in the Dockerfile + entrypoint:**
- Dockerfile runtime stage now sets `CUDA_HOME=/usr/local/cuda` and puts
  `/usr/local/cuda/bin` on `PATH` unconditionally, and adds a **build-time
  assertion** `RUN nvcc --version && test -f $CUDA_HOME/include/cuda_runtime.h` so a
  toolkit-less image can never be published again (the build fails loudly instead).
- `entrypoint.sh` NVFP4 block now installs the CUDA **math + cudnn dev** packages
  (`cuda-nvcc-12-8`, `cuda-cudart-dev-12-8`, `cuda-libraries-dev-12-8`,
  `libcudnn9-dev-cuda-12`, `ninja-build`) and runs
  `apt-mark unhold libcublas-12-8 libcudnn9-cuda-12` before building TE.

These need validation on the next CI image build + pod boot.

## RunPod / SSH notes

- **Custom Docker images can't use RunPod proxy SSH** (`ssh.runpod.io`) — it
  returns `container not found`. Use **direct TCP SSH**: create the pod with
  `supportPublicIp: true` and ports `["22/tcp", "8000/http"]`; RunPod assigns a
  public IP + mapped port. (Do NOT override `dockerEntrypoint` — it breaks the
  image's `entrypoint.sh`; only override `dockerStartCmd` if you must.)
- GPU type id: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (96GB, ~$1.69/hr,
  COMMUNITY). The 5090 was unavailable (low stock).
- Models download to `~/.daydream-scope/models` unless `DAYDREAM_SCOPE_MODELS_DIR`
  is exported in the SSH session (the container env var is NOT inherited by sshd
  sessions). Keep server + download using the same dir.
- All `longlive2` model repos are **public on HF — no HF token needed**.

## Secrets (rotate after use)

- Cloudflare TURN keys (validated working) were set on the pod env only, not in
  git/memory. **Rotate the TURN key and the RunPod API key** that were used.

## Re-setup runbook (for a fresh pod tomorrow)

1. Create pod: image `zmiccer/scope_persona:latest`, GPU
   `NVIDIA RTX PRO 6000 Blackwell Server Edition`, COMMUNITY, `supportPublicIp:true`,
   ports `["22/tcp","8000/http"]`, volume 60GB `/workspace`, env:
   `ATTENTION_BACKEND=flash`, `DAYDREAM_SCOPE_MODELS_DIR=/root/.daydream-scope/models`,
   `PUBLIC_KEY=<your ed25519 pubkey>`. `dockerStartCmd` = install/run `sshd` +
   `sleep infinity` (do not override `dockerEntrypoint`).
2. SSH in via the public IP + mapped port (`-i ~/.ssh/id_ed_runpod`).
3. Overlay branch code: `git clone -b longlive2 https://github.com/zzmicer/scope_persona.git`
   and replace `/app/src/scope` with the clone's `src/scope`; add
   `echo /app/src > <site-packages>/__aaa_scope_src.pth` so `scope` imports and
   the `_startup` patch runs (the image's scope install is path-based).
4. Download weights: `uv run --no-sync download_models --pipeline longlive2`
   (pulls Wan2.2-TI2V-5B + umt5 fp8 text encoder + LongLive-2.0-5B-NVFP4-S2).
   For BF16 also: `hf_hub_download('Efficient-Large-Model/LongLive-2.0-5B',
   'model_bf16.pt', local_dir='.../LongLive-2.0-5B')` (~9.4GB).
5. **For BF16 (working today):** launch the server with the Cloudflare TURN env
   vars and load `longlive2` with `load_params={"precision":"bf16"}`. With the
   committed schedule fix, the default frontend connect renders correctly.
6. **For NVFP4 (TODO):** just set `LONGLIVE2_NVFP4=1` in the pod env — the
   `entrypoint.sh` NVFP4 block now auto-installs the CUDA/cuDNN dev headers (after
   `apt-mark unhold`), builds TE, upgrades `transformers>=5`, installs `fouroversix`,
   and compiles `kv_dequant` (`LONGLIVE_KV_DEQUANT_ARCHS=120a`). Remaining
   validation: confirm `nvfp4-s2` renders, measure fps, and confirm the umt5 text
   encoder tolerates transformers v5.

## Next steps

- [x] Resolve `fouroversix` ↔ `transformers` version conflict — needs
      `transformers>=5.0.0` (`WeightConverter`); entrypoint upgrades it in the
      NVFP4-only branch. **Still:** finish TE build + validate `nvfp4-s2` render +
      measure fps on the pod.
- [x] Confirm the real 5B **4-step** DMD timesteps from upstream and set them in
      `model.yaml` → `[1000, 946, 854, 681]` (was wrong `[1000,750,500,250]`).
      **Still:** the **2-step** s2 schedule is unverified (upstream ships no
      `sampling_steps:2` config); current subsample = `[1000, 681]`.
- [x] Fix the Dockerfile/CI so the runtime image contains the CUDA toolkit
      (CUDA_HOME + PATH + build-time `nvcc` assertion), and have `entrypoint.sh`
      install the dev headers it needs for the NVFP4 build. **Still:** verify on the
      next CI build + pod boot.
- [ ] Decide whether the frontend should stop sending a hardcoded
      `denoising_step_list` for longlive2 (backend override now guards it anyway).
