"""
Overlap the SR round-trip with generation, and fix the WS meta size.

As first deployed, SR was a blocking call inside the chunk loop, so wall time per
chunk was generator + SR = 2.97s against a 2.0s budget (0.67x realtime). Nothing
there was compute-bound: GPU 1 idled while GPUs 2/3 generated and vice versa.
Submitting chunk N and emitting chunk N-1 makes wall time max(gen, SR).

Cost: one chunk (2.0s) of extra end-to-end latency, and audio is held back with
its own video chunk -- the demo drives A/V sync off frames_emitted /
audio_samples_emitted, so releasing audio early would desync the lips.

Deliberately no `continue` in the loop: transition_ttl is decremented
independently on BOTH ranks, so a rank-0-only skip would give the two ranks
different prompts and diverge the stream.

Idempotent; backup at .bak3. Requires add_sr.py to have run first.
"""

import shutil
import sys

PATH = "/workspace/soulx/SoulX-LiveAct/interactive_demo.py"

A_META = '''                    "w": engine.width,
                    "h": engine.height,'''
P_META = '''                    "w": engine.out_width,
                    "h": engine.out_height,'''

A_IMPORT = '''SIZE_ALIGN = 16'''
P_IMPORT = '''# one worker only: the sidecar serializes on its own lock, and two chunks in
# flight would reorder through its streaming caches
_sr_pool = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sr")

SIZE_ALIGN = 16'''

A_CF = '''import queue'''
P_CF = '''import concurrent.futures as _cf
import queue'''

A_SUB = '''                        vids = None
                        sr_u8 = _sr_request(
                            self.sr_url, latent, cur_ctx, self.rank
                        )'''
P_SUB = '''                        vids = None
                        sr_u8 = None
                        sr_job = (
                            (
                                latent.detach().to(torch.bfloat16).cpu(),
                                [c.detach().to(torch.bfloat16).cpu() for c in cur_ctx],
                            )
                            if self.rank == 0
                            else None
                        )'''

A_EMIT = '''                if self.rank == 0:
                    if sr_u8 is not None:
                        arr = sr_u8'''
P_EMIT = '''                n_emit = 0
                if self.rank == 0 and self.sr_url:
                    # submit chunk N, emit chunk N-1 -> SR overlaps the next denoise
                    sr_inflight.append(
                        (_sr_pool.submit(_sr_request, self.sr_url,
                                         sr_job[0], sr_job[1], 0), heard)
                    )
                    if len(sr_inflight) >= 2:
                        fut0, heard = sr_inflight.pop(0)
                        sr_u8 = fut0.result()
                    else:
                        sr_u8 = None

                if self.rank == 0 and (sr_u8 is not None or not self.sr_url):
                    if sr_u8 is not None:
                        arr = sr_u8'''

A_STATE = '''        pre_latent = None
        sr_u8 = None
        if self.sr_url and self.rank == 0:
            _sr_reset(self.sr_url)
        iteration = 0'''
P_STATE = '''        pre_latent = None
        sr_u8 = None
        sr_job = None
        sr_inflight = []
        if self.sr_url and self.rank == 0:
            _sr_reset(self.sr_url)
        iteration = 0'''

PATCHES = [
    ("meta size", A_META, P_META),
    ("cf import", A_CF, P_CF),
    ("pool", A_IMPORT, P_IMPORT),
    ("submit", A_SUB, P_SUB),
    ("emit", A_EMIT, P_EMIT),
    ("state", A_STATE, P_STATE),
]


def main():
    src = open(PATH).read()
    if "_sr_pool" in src:
        print("already patched")
        return 0
    if "_sr_request" not in src:
        print("base SR patch missing - run add_sr.py first")
        return 1
    for name, anchor, _ in PATCHES:
        if src.count(anchor) != 1:
            print(f"ANCHOR {name}: found {src.count(anchor)}x, need 1")
            return 1
    shutil.copy(PATH, PATH + ".bak3")
    for name, anchor, patch in PATCHES:
        src = src.replace(anchor, patch, 1)
    open(PATH, "w").write(src)
    print("async patch ok (backup at .bak3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
