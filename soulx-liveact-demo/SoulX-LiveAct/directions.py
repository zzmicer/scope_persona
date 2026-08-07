"""Pull stage directions out of what the brain says.

The character answers in prose with the motion written inline, in brackets,
optionally with how long it lasts:

    Aww, thank you! [she turns around slowly, 2s] Do you like the back?

That line has to split three ways: the speech goes to TTS, the motion goes to
the video generator as a transient transition, and "2s" becomes the TTL that
transition is held for. This module is that split, and nothing else.

Why brackets and not the JSON schema this used to ask for: measured on the live
demo, Qwen2.5-1.5B returned `action: null` on 10 of 10 "wave at me" turns and
wrote the motion into `say` instead — so the pipeline got nothing and the TTS
read the stage direction aloud as dialogue. Small instruct models are far more
reliable at narrating a motion inline than at filling a nullable JSON field, and
when they do slip back into either old habit (bare JSON, or a bare third-person
sentence) the parser below recovers the motion rather than speaking it.

Deliberately stdlib-only, like `stage.py`, so it can be tested without the model.
"""

import json
import math
import re
from dataclasses import dataclass

# One generated chunk is 32 frames @16fps. A duration in the script is only ever
# realizable to chunk granularity, so 2s is the floor and the quantum.
CHUNK_SECONDS = 2.0
MAX_HOLD_CHUNKS = 120  # ~4min; past this it is a pose, not a gesture

# Verbs that mean the body ends up somewhere new. These make a motion sustained:
# the resulting posture becomes the resting state instead of expiring back to
# idle. Everything else (wave, nod, wink) is momentary by default.
POSTURE_PATTERNS = (
    r"sit(s|ting)?\b",
    # The model rarely writes the bare verb. Observed live: "She takes a seat on
    # the couch, relaxing." — a sit-down that the `sit` pattern alone misses, so
    # the one posture this feature exists for failed to stick.
    r"(take|takes|taking|have|has|having)\s+a\s+seat",
    r"(settle|settles|settling|perch|perches|perching)\b",
    r"(curl|curls|curling)\s+up",
    r"stand(s|ing)?\b",
    r"(lie|lies|lying|lay|lays|laying)\b",
    r"kneel(s|ing)?\b",
    r"crouch(es|ing)?\b",
    r"squat(s|ting)?\b",
    r"lean(s|ing)?\b",
    r"turn(s|ing)?\s+(around|away|her back)",
    r"(cross|crosses|crossing)\s+her\s+arms",
    r"(get|gets|getting)\s+up",
    r"(face|faces|facing)\s+(away|the camera)",
)
POSTURE_RE = re.compile("|".join(POSTURE_PATTERNS), re.I)

# An explicit "and stays like that" beats the posture heuristic in both
# directions: it can make a gesture stick, and its absence plus a duration keeps
# a posture change temporary ("turn around for 2s" -> she turns back).
STAYS_RE = re.compile(
    r"\b(stay|stays|staying|remain|remains|remaining|holds? it)\b", re.I
)

MOTION_WORDS = (
    "wave waves waving nod nods nodding smile smiles smiling grin grins laugh "
    "laughs laughing giggle giggles giggling wink winks winking turn turns "
    "turning sit sits sitting stand stands standing lie lies lying kneel kneels "
    "lean leans leaning cross crosses raise raises raising lift lifts point "
    "points pointing dance dances dancing jump jumps spin spins spinning bow "
    "bows stretch stretches shrug shrugs clap claps tilt tilts blows hug hugs "
    "step steps walk walks runs glance glances covers places puts holds twirl "
    "twirls curtsy salutes poses blushes leans reaches touches shakes claps "
    "flops sways rocks tosses brushes adjusts"
).split()
MOTION_RE = re.compile(r"\b(" + "|".join(sorted(set(MOTION_WORDS))) + r")\b", re.I)

DURATION_RE = re.compile(
    r"[,;]?\s*\b(?:for\s+|about\s+|~)*"
    r"(\d+(?:\.\d+)?)\s*"
    r"(seconds?|secs?|s|minutes?|mins?|m)\b\.?",
    re.I,
)

BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
# Roleplay models reach for *asterisks* as often as brackets. Require two words
# so a single emphasized word ("*so* good") is left in the speech.
ASTERISK_RE = re.compile(r"\*{1,2}\s*([^*\n]*?\s+[^*\n]*?)\s*\*{1,2}")

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

# First person slips through often enough to be worth fixing: the video prompt
# is a third-person visual description, and "I turn around" conditions it badly.
IRREGULAR = {"am": "is", "have": "has", "do": "does", "go": "goes", "can": "can"}


# ----------------------------------------------------------------------------
# The prompt that produces the format lives next to the parser that consumes it,
# so the two cannot drift and a harness can test the real thing without
# importing the model stack.
# ----------------------------------------------------------------------------


def system_prompt(name, persona):
    return (
        f"You are {name}, an anime character appearing live on video. "
        f"Your personality is: {persona} "
        "Chat naturally with the user, in English, using 1-2 short sentences.\n"
        "You have a body on camera, and you MOVE IT YOURSELF. Write every motion "
        "inline, in square brackets, as a short third-person visual description "
        "ending with how long it lasts:\n"
        "  [she waves her hand at the camera, 2s]\n"
        "  [she turns around slowly, 3s]\n"
        "Leave the duration off ONLY for a posture you mean to KEEP (sit down, "
        "stand up, lie down, kneel, lean back, cross arms) — with no duration you "
        "stay that way until something changes it:\n"
        "  [she sits down on the floor, cross-legged]\n"
        "ALWAYS say something as well: never reply with a motion alone. Never "
        "describe your motion outside the brackets — text outside them is read "
        "out loud in your voice.\n"
        "Move in almost every reply, even when the user did not ask for it — "
        "react, gesture while you talk, use your body."
    )


# Format is taught by example, not by description: a 1.5B model reliably copies a
# shape it has just seen and reliably ignores a schema it was only told about.
# Measured against the live aux LLM: the old JSON prompt fired an action on 6 of
# 20 motion requests, these examples plus the prompt above on 20 of 20.
FEWSHOT = [
    {"role": "user", "content": "hey! how are you today?"},
    {
        "role": "assistant",
        "content": "I'm great, thanks for asking! [she waves at the camera, 2s] "
        "What have you been up to?",
    },
    {"role": "user", "content": "you are so good, can you turn around?"},
    {
        "role": "assistant",
        "content": "Sure, take a look! [she turns around slowly, 3s] Ta-da!",
    },
    {"role": "user", "content": "you can sit down if you want"},
    {
        "role": "assistant",
        "content": "Thanks, my feet were getting tired! "
        "[she sits down on the floor, cross-legged]",
    },
    {"role": "user", "content": "what did you do this morning?"},
    {
        "role": "assistant",
        "content": "I watched the rain from the window for ages. "
        "[she tilts her head and smiles softly, 2s] It was so peaceful.",
    },
]


@dataclass
class Direction:
    """One parsed reply.

    say    — the dialogue, stage directions removed; "" means she acts silently.
    action — transient motion for the generator, or None.
    pose   — new sustained state; None leaves the current one alone, "" clears it.
    hold   — chunks to hold `action`, or None to use the server default.
    script — canonical `say [action]` form, fed back into chat history so the
             model keeps seeing its own format.
    """

    say: str
    action: str = None
    pose: str = None
    hold: int = None
    script: str = ""


def seconds_to_chunks(seconds):
    """Round a scripted duration UP to whole chunks; a 1s beat still renders."""
    if seconds is None or seconds <= 0:
        return None
    return max(1, min(MAX_HOLD_CHUNKS, math.ceil(seconds / CHUNK_SECONDS)))


def _take_duration(text):
    """-> (text without the duration phrase, seconds or None)."""
    match = DURATION_RE.search(text)
    if not match:
        return text, None
    value = float(match.group(1))
    unit = match.group(2).lower()
    seconds = value * 60.0 if unit.startswith("m") and unit != "s" else value
    return (text[: match.start()] + " " + text[match.end() :]), seconds


def _tidy(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+([,.!?;:…])", r"\1", text)
    text = re.sub(r"([,;:])\s*([.!?])", r"\2", text)
    return text.strip(" ,;:-")


def _to_third_person(text, name):
    """ "I stretch my arms" -> "She stretches her arms". Third person is left be."""
    match = re.match(r"^\s*I\s+(\w+)(.*)$", text, re.S)
    if not match:
        return re.sub(r"^\s*(i|I)'m\b", "She is", text)
    verb, rest = match.group(1), match.group(2)
    low = verb.lower()
    if low in IRREGULAR:
        verb = IRREGULAR[low]
    elif low.endswith(("s", "sh", "ch", "x", "z", "o")):
        verb = low + "es"
    elif low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        verb = low[:-1] + "ies"
    elif not low.endswith("ing") and not low.endswith("ed"):
        verb = low + "s"
    # The rest of a first-person line owns its pronouns too, and "she stretches
    # my arms" would condition the generator on a second body.
    rest = re.sub(r"\bmyself\b", "herself", rest, flags=re.I)
    rest = re.sub(r"\bmy\b", "her", rest, flags=re.I)
    rest = re.sub(r"\bme\b", "her", rest, flags=re.I)
    rest = re.sub(r"\bI\b", "she", rest)
    return f"She {verb}{rest}"


def _looks_like_direction(sentence, name):
    """A bare third-person motion sentence in the speech slot is a stage
    direction the model forgot to bracket — the exact 1.5B failure mode."""
    lead = r"(she|her|" + (re.escape(name) + r"|" if name else "") + r"the girl)"
    if not re.match(rf"^\s*{lead}\b", sentence, re.I):
        return False
    return bool(MOTION_RE.search(sentence))


def _classify(action, had_duration):
    """-> pose text or None. A posture change with no stated duration is where
    she stays; with one, she goes back."""
    if not action:
        return None
    if STAYS_RE.search(action):
        return action
    if had_duration:
        return None
    return action if POSTURE_RE.search(action) else None


def _from_json(raw):
    """The old schema, still honoured if a bigger/remote model emits it.

    Tolerates the concatenated `{"say":..}\\n{"action":..}` the 1.5B liked to
    produce, which the old single-slice parse turned into a wall of raw text
    spoken aloud.
    """
    merged = {}
    decoder = json.JSONDecoder()
    index = 0
    while index < len(raw):
        start = raw.find("{", index)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(raw[start:])
        except ValueError:
            index = start + 1
            continue
        if isinstance(obj, dict):
            for key, value in obj.items():
                if merged.get(key) in (None, "", []):
                    merged[key] = value
        index = start + end
    return merged or None


def _field(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    return None if value.lower() in ("", "null", "none", "n/a") else value


def parse(raw, name="", default_hold=None):
    """Split a raw model reply into speech + motion.

    `default_hold` (chunks) is only reported back when the script states no
    duration of its own, so the caller can tell "she said 2s" from "she said
    nothing and you chose 16s".
    """
    raw = (raw or "").strip()
    if not raw:
        return Direction(say="")

    action = pose = None
    seconds = None

    payload = _from_json(raw) if "{" in raw and '"' in raw else None
    if payload and ("say" in payload or "action" in payload or "pose" in payload):
        speech = _field(payload.get("say")) or ""
        action = _field(payload.get("action"))
        pose = _field(payload.get("pose"))
        try:
            seconds = (
                float(payload.get("seconds") or payload.get("duration") or 0) or None
            )
        except (TypeError, ValueError):
            seconds = None
    else:
        speech = raw

    # Bracketed and asterisked directions, in the order they appear. Multiple
    # ones get joined: the generator takes a single prompt, and two motions in
    # one turn read as one continuous move.
    found = []

    def _capture(match):
        found.append(match.group(1))
        return " "

    speech = BRACKET_RE.sub(_capture, speech)
    speech = ASTERISK_RE.sub(_capture, speech)

    # Whatever is left may still be a bare stage direction the model forgot to
    # mark up. Promote leading third-person motion sentences; keep real speech.
    if not found:
        kept = []
        for sentence in SENTENCE_SPLIT_RE.split(speech):
            if sentence.strip() and _looks_like_direction(sentence, name):
                found.append(sentence)
            else:
                kept.append(sentence)
        speech = " ".join(kept)

    if found:
        parts = []
        for item in found:
            item, item_seconds = _take_duration(item)
            if item_seconds and not seconds:
                seconds = item_seconds
            item = _tidy(_to_third_person(_tidy(item), name))
            if item:
                parts.append(item[0].upper() + item[1:])
        merged = ". ".join(p.rstrip(".") for p in parts)
        if merged:
            merged += "."
            action = merged if not action else f"{action.rstrip('.')}. {merged}"

    if action:
        action, action_seconds = _take_duration(action)
        seconds = seconds or action_seconds
        action = _tidy(action)
        if pose is None:
            pose = _classify(action, seconds is not None)

    say = _tidy(speech)
    # Speech that is only leftover punctuation or a quote wrapper is not speech.
    say = say.strip("\"“”' ")

    hold = seconds_to_chunks(seconds) or default_hold
    script = say
    if action:
        script = f"{say} [{action}]".strip() if say else f"[{action}]"
    return Direction(say=say, action=action, pose=pose, hold=hold, script=script)
