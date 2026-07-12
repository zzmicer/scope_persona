"""Text command -> camera trajectory synthesis for LingBot-World-V2.

LingBot-World-V2's released inference conditions the DiT on per-latent-frame
framewise camera deltas (as Plücker embeddings). This module turns free-text
commands like "walk forward", "turn left", "orbit around her" into absolute
camera-to-world pose tracks sampled at the latent frame rate (4 video frames
per latent frame, 16 fps video -> 4 latent frames per second).

Conventions (OpenCV camera): +x right, +y down, +z forward.
Translation magnitudes are chosen to match the training-time normalization
(framewise deltas normalized so the fastest frame has norm ~1): typical
per-latent-frame steps in the released examples are ~0.1-0.9.
"""

from dataclasses import dataclass

import numpy as np

LATENT_FPS = 4.0  # 16 fps video / 4 frames per latent

# per-latent-frame translation step by speed
_SPEEDS = {"slow": 0.15, "normal": 0.4, "fast": 0.8}
# per-latent-frame rotation in degrees by speed
_TURN_DEG = {"slow": 2.0, "normal": 4.5, "fast": 8.0}


@dataclass
class Motion:
    kind: str  # forward|back|left|right|up|down|turn_left|turn_right|look_up|look_down|orbit_left|orbit_right|stay
    seconds: float = 2.0
    speed: str = "normal"


_MOTION_KEYWORDS = [
    # (kind, keywords) — first match wins, so put multiword phrases first
    (
        "orbit_left",
        ["orbit left", "circle left", "orbit around", "circle around", "orbit"],
    ),
    ("orbit_right", ["orbit right", "circle right"]),
    ("turn_left", ["turn left", "rotate left", "turn around"]),
    ("turn_right", ["turn right", "rotate right"]),
    ("look_up", ["look up", "tilt up"]),
    ("look_down", ["look down", "tilt down"]),
    ("left", ["strafe left", "move left", "step left", "go left", "slide left"]),
    ("right", ["strafe right", "move right", "step right", "go right", "slide right"]),
    ("back", ["back", "backward", "retreat", "move away", "step away"]),
    ("up", ["fly up", "rise", "ascend", "move up"]),
    ("down", ["descend", "move down", "lower"]),
    (
        "forward",
        [
            "forward",
            "ahead",
            "walk",
            "approach",
            "closer",
            "advance",
            "towards",
            "toward",
        ],
    ),
    ("stay", ["stay", "stop", "stand still", "wait", "hold", "idle"]),
]


def parse_command(text: str) -> tuple[list[Motion], str | None]:
    """Parse a user command into (motions, event_prompt).

    Motion keywords produce camera moves. Text with no motion keyword — or
    with an explicit "prompt:"/"event:"/"scene:" prefix — becomes an event
    prompt that re-conditions the model (e.g. "she smiles and waves").
    """
    raw = text.strip()
    lowered = raw.lower()

    for prefix in ("prompt:", "event:", "scene:"):
        if lowered.startswith(prefix):
            return [], raw[len(prefix) :].strip()

    seconds = 2.0
    words = lowered.replace(",", " ").split()
    for i, w in enumerate(words):
        if w in ("second", "seconds", "sec", "secs", "s") and i > 0:
            try:
                seconds = float(words[i - 1])
            except ValueError:
                pass

    speed = "normal"
    if any(w in lowered for w in ("slow", "slowly", "gently")):
        speed = "slow"
    if any(w in lowered for w in ("fast", "quickly", "run", "sprint")):
        speed = "fast"

    motions = []
    remaining = lowered
    # allow simple compounds: "turn left and walk forward"
    for part in remaining.split(" and "):
        for kind, keywords in _MOTION_KEYWORDS:
            if any(k in part for k in keywords):
                motions.append(Motion(kind=kind, seconds=seconds, speed=speed))
                break

    if not motions:
        return [], raw
    return motions, None


def _yaw(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    m = np.eye(4)
    m[0, 0] = np.cos(r)
    m[0, 2] = np.sin(r)
    m[2, 0] = -np.sin(r)
    m[2, 2] = np.cos(r)
    return m


def _pitch(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    m = np.eye(4)
    m[1, 1] = np.cos(r)
    m[1, 2] = -np.sin(r)
    m[2, 1] = np.sin(r)
    m[2, 2] = np.cos(r)
    return m


def _trans(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m


def _step_delta(kind: str, speed: str) -> np.ndarray:
    """One latent-frame camera delta (in the current camera's frame)."""
    t = _SPEEDS[speed]
    r = _TURN_DEG[speed]
    if kind == "forward":
        return _trans(0, 0, t)
    if kind == "back":
        return _trans(0, 0, -t)
    if kind == "left":
        return _trans(-t, 0, 0)
    if kind == "right":
        return _trans(t, 0, 0)
    if kind == "up":
        return _trans(0, -t, 0)  # +y is down
    if kind == "down":
        return _trans(0, t, 0)
    if kind == "turn_left":
        return _yaw(-r)
    if kind == "turn_right":
        return _yaw(r)
    if kind == "look_up":
        return _pitch(-r)
    if kind == "look_down":
        return _pitch(r)
    if kind == "orbit_left":
        # translate sideways while counter-yawing toward a pivot ~3 units ahead
        arc = np.deg2rad(r) * 3.0
        return _trans(-arc, 0, 0) @ _yaw(r)
    if kind == "orbit_right":
        arc = np.deg2rad(r) * 3.0
        return _trans(arc, 0, 0) @ _yaw(-r)
    if kind == "stay":
        return np.eye(4)
    raise ValueError(f"unknown motion kind: {kind}")


def trajectory_from_motions(
    motions: list[Motion],
    start_c2w: np.ndarray,
    chunk_size: int = 4,
) -> np.ndarray:
    """Build an absolute c2w pose track at the latent frame rate.

    Returns [n, 4, 4] with n a positive multiple of chunk_size; the track
    continues from (but does not include) start_c2w. Padding to the chunk
    boundary repeats the final motion's delta so movement stays smooth.
    """
    deltas = []
    for m in motions:
        n = max(1, round(m.seconds * LATENT_FPS))
        deltas.extend([_step_delta(m.kind, m.speed)] * n)

    pad = (-len(deltas)) % chunk_size
    pad_delta = deltas[-1] if deltas else np.eye(4)
    deltas.extend([pad_delta] * pad)

    track = []
    cur = start_c2w.copy()
    for d in deltas:
        cur = cur @ d
        track.append(cur.copy())
    return np.stack(track)


def default_intrinsics(width: int, height: int) -> np.ndarray:
    """Per-frame [fx, fy, cx, cy] matching the released examples' FOV,
    already scaled for a (width, height) render target."""
    return np.array(
        [415.53 * width / 832.0, 415.69 * height / 480.0, width / 2.0, height / 2.0],
        dtype=np.float32,
    )
