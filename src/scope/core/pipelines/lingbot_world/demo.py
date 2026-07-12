"""Interactive text-controlled world demo for LingBot-World-V2.

Pass a start image and steer the world with text: motion commands move the
camera ("walk forward", "turn left", "orbit around her", "look up"), any
other text becomes a scene/event prompt ("she smiles and waves at the
camera"). Each turn generates ~2s of video; the session mp4 is rewritten
after every turn so partial results always exist.

Run on a GPU box with the upstream repo + checkpoints available:

    python demo.py \
        --lingbot-repo /workspace/lingbot-world-v2 \
        --ckpt-dir /workspace/lingbot-world-v2-14b-causal-fast \
        --image girl.jpg \
        --prompt "A young woman stands in a sunlit park, looking at the camera" \
        --out /workspace/lingbot_out/session.mp4 \
        [--script commands.txt]   # one command per line; otherwise reads stdin
"""

import argparse
import logging
import os
import sys
import time


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lingbot-repo", default=os.environ.get("LINGBOT_WORLD_REPO"))
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default="session.mp4")
    p.add_argument("--script", default=None, help="file with one command per line")
    p.add_argument("--max-frames", type=int, default=321)
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--local-attn-size", type=int, default=18)
    p.add_argument("--sink-size", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", default="480*832", help="max_area as H*W")
    args = p.parse_args()
    assert args.lingbot_repo, "--lingbot-repo or LINGBOT_WORLD_REPO required"
    return args


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    sys.path.insert(0, args.lingbot_repo)

    import wan
    from PIL import Image
    from wan.configs import WAN_CONFIGS

    from scope.core.pipelines.lingbot_world.actions import (
        parse_command,
        trajectory_from_motions,
    )
    from scope.core.pipelines.lingbot_world.session import LingbotWorldSession

    h, w = (int(x) for x in args.size.split("*"))

    logging.info("loading pipeline (this takes ~1 min)...")
    t0 = time.perf_counter()
    pipe = wan.WanI2VCausal(
        config=WAN_CONFIGS["i2v-A14B"],
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        init_on_cpu=False,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        infer_mode="causal_fast",
    )
    logging.info("pipeline loaded in %.0fs", time.perf_counter() - t0)

    img = Image.open(args.image).convert("RGB")
    session = LingbotWorldSession(
        pipe,
        img,
        args.prompt,
        max_frames=args.max_frames,
        chunk_size=args.chunk_size,
        max_area=h * w,
        seed=args.seed,
    )
    logging.info(
        "session ready: %dx%d, budget %d latent frames (~%.0fs of video)",
        session.w,
        session.h,
        session.lat_f_max,
        session.lat_f_max / 4.0,
    )

    if args.script:
        with open(args.script) as f:
            commands = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
        source = iter(commands)
    else:
        source = None
        print(
            "\nType commands (motion: 'walk forward', 'turn left', 'orbit', "
            "'look up', 'stay'; anything else = event prompt; 'quit' to finish):"
        )

    prompt_is_event = False
    while True:
        if source is not None:
            cmd = next(source, None)
            if cmd is None:
                break
            print(f"\n>>> {cmd}")
        else:
            try:
                cmd = input("\n>>> ").strip()
            except EOFError:
                break
            if not cmd:
                continue
        if cmd.lower() in ("quit", "exit", "q"):
            break

        motions, event = parse_command(cmd)
        if event is not None:
            # Compose with the base world prompt so the world doesn't morph
            # into the event's subject; events are transient (one turn).
            session.set_prompt(f"{args.prompt} {event}")
            prompt_is_event = True
            print(f"    [event] prompt -> {event!r}")
            # events still need frames to unfold: idle the camera for 2s
            from scope.core.pipelines.lingbot_world.actions import Motion

            motions = [Motion(kind="stay", seconds=2.0)]
        else:
            if prompt_is_event:
                session.set_prompt(args.prompt)
                prompt_is_event = False
            print(f"    [motion] {[m.kind for m in motions]}")

        track = trajectory_from_motions(
            motions, session.cur_c2w, chunk_size=session.chunk_size
        )
        if len(track) > session.latent_budget_left:
            print(
                f"    [warn] budget left {session.latent_budget_left} latent frames; trimming"
            )
            track = track[: session.latent_budget_left]
            if len(track) == 0:
                print("    [done] latent budget exhausted")
                break

        t0 = time.perf_counter()
        frames = session.step(track)
        dt = time.perf_counter() - t0
        session.save(args.out)
        print(
            f"    [ok] +{frames.shape[1]} frames ({frames.shape[1] / 16.0:.1f}s video) "
            f"in {dt:.1f}s | total {session.frames_generated} frames | saved {args.out}"
        )
        if session.latent_budget_left == 0:
            print("    [done] latent budget exhausted")
            break

    if session.frames:
        session.save(args.out)
        print(f"\nfinal video: {args.out} ({session.frames_generated} frames)")


if __name__ == "__main__":
    main()
