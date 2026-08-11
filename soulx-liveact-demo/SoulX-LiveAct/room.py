"""Presence and open chat for the crowd around one live stage.

`stage.py` owns the seat -- who may drive the character. This owns everything
the crowd does that costs no GPU: who is here, what they are saying to each
other, and the "37 watching" that makes a rough demo feel alive.

The split matters because the two have opposite access rules. Driving is
exclusive and token-gated; talking is open to everyone, always. Keeping them in
separate objects is what stops "can they chat?" and "can they drive?" from
collapsing into one confused permission.

There is deliberately no separate "suggestion" concept. A watcher's message IS
the suggestion: the driver's client renders every room message with a tap
target that replays it as their own action. One list, one mental model, and
watchers do not have to choose which box to type in.

Messages carry monotonic ids so a client can ask for "everything after 41"
rather than re-reading the whole log on a 2s timer.

Stdlib-only and free of Flask/torch, like its sibling, so the whole social layer
is testable on a laptop with no model loaded. Time is injected for the same
reason: every rule here is a deadline, and tests should not sleep.
"""

import threading
from collections import deque
from dataclasses import asdict, dataclass

from stage import sanitize_name

MAX_TEXT_LEN = 240
DEFAULT_HISTORY = 200
# A viewer who stopped polling is gone. This only has to exceed the client's
# poll interval; it is the same liveness idea as the stage's idle timeout, but
# it may be more generous because being counted as present costs nothing.
DEFAULT_PRESENCE_TIMEOUT = 45.0
# Cheapest possible flood control: a floor on the gap between one client's
# messages. Not moderation -- that is a separate concern on the way to the
# model -- just enough that one bored tab cannot scroll the room away.
DEFAULT_MIN_INTERVAL = 1.0
DEFAULT_BURST = 4


class RoomError(Exception):
    status = 400

    def __init__(self, message, status=None):
        super().__init__(message)
        self.message = message
        if status is not None:
            self.status = status


class TooFast(RoomError):
    status = 429


@dataclass
class Message:
    id: int
    kind: str  # "chat" (a person) | "persona" (her) | "system" (the room)
    who: str
    text: str
    t: float
    # Present only on "chat": lets the driver's client replay someone else's
    # line as an action without the server having to remember who suggested it.
    client_id: str = ""


@dataclass
class _Present:
    name: str
    seen: float
    # Token-bucket allowance for posting, refilled at 1/MIN_INTERVAL per second.
    tokens: float = DEFAULT_BURST
    last_post: float = 0.0


class Room:
    """Who is here and what they are saying. One room, one stage, one GPU."""

    def __init__(
        self,
        clock=None,
        history=DEFAULT_HISTORY,
        presence_timeout=DEFAULT_PRESENCE_TIMEOUT,
        min_interval=DEFAULT_MIN_INTERVAL,
        burst=DEFAULT_BURST,
    ):
        self._clock = clock or __import__("time").time
        self._lock = threading.RLock()
        self._messages = deque(maxlen=history)
        self._next_id = 1
        self._present = {}  # client_id -> _Present
        self.presence_timeout = presence_timeout
        self.min_interval = min_interval
        self.burst = burst

    # -- presence ----------------------------------------------------------

    def see(self, client_id, name=None):
        """Mark a client present. Every poll is a presence ping, so the roster
        stays honest without a second heartbeat endpoint."""
        if not client_id:
            raise RoomError("client_id required")
        now = self._clock()
        with self._lock:
            entry = self._present.get(client_id)
            if entry is None:
                # Seed the allowance from THIS room's burst, not the dataclass
                # default -- otherwise a room configured tighter than the
                # default silently hands every newcomer the default's credit.
                entry = _Present(name=sanitize_name(name), seen=now, tokens=self.burst)
                self._present[client_id] = entry
            else:
                entry.seen = now
                if name is not None:
                    entry.name = sanitize_name(name)
            return entry.name

    def leave(self, client_id):
        with self._lock:
            self._present.pop(client_id, None)

    def watching(self):
        """How many are here now. Sweeps the disappeared as it counts -- the
        roster, like the queue, only has to be right when someone looks."""
        now = self._clock()
        with self._lock:
            gone = [
                cid
                for cid, e in self._present.items()
                if (now - e.seen) > self.presence_timeout
            ]
            for cid in gone:
                del self._present[cid]
            return len(self._present)

    def name_of(self, client_id):
        with self._lock:
            entry = self._present.get(client_id)
            return entry.name if entry else "guest"

    # -- talking -----------------------------------------------------------

    def post(self, client_id, text, name=None):
        """A person says something to the room. Open to everyone, driver or not."""
        body = " ".join(str(text or "").split())[:MAX_TEXT_LEN]
        body = "".join(ch for ch in body if ch.isprintable())
        if not body:
            raise RoomError("empty message")
        display = self.see(client_id, name)
        now = self._clock()
        with self._lock:
            entry = self._present[client_id]
            # Refill before spending so a quiet client is not punished for the
            # gap: allowance accrues at exactly the sustainable rate.
            if entry.last_post:
                entry.tokens = min(
                    self.burst,
                    entry.tokens + (now - entry.last_post) / self.min_interval,
                )
            if entry.tokens < 1.0:
                raise TooFast("slow down")
            entry.tokens -= 1.0
            entry.last_post = now
            return self._append("chat", display, body, client_id=client_id)

    def said(self, who, text, client_id=""):
        """Mirror a line that was already authorized at another gate.

        The driver's message to her goes through the turn token, not through
        here, so charging it against the flood allowance a second time would
        let a long turn silently throttle the driver out of their own seat.
        Watchers still need to see it: her replies are incomprehensible without
        the prompts that caused them.
        """
        body = " ".join(str(text or "").split())[:MAX_TEXT_LEN]
        body = "".join(ch for ch in body if ch.isprintable())
        with self._lock:
            return self._append("chat", sanitize_name(who), body, client_id=client_id)

    def persona(self, text, who="Chano"):
        """Something she said, mirrored into the room so watchers follow along."""
        with self._lock:
            return self._append("persona", who, str(text or "").strip())

    def system(self, text):
        """Room narration: turns starting and ending, outfit rituals."""
        with self._lock:
            return self._append("system", "", str(text or "").strip())

    # -- views -------------------------------------------------------------

    def since(self, after_id):
        """Messages newer than `after_id`, oldest first.

        A client that has fallen further behind than the buffer is deep simply
        gets what is left: the room is a conversation, not a ledger, and
        replaying an hour of backlog into a 2s poll would be worse than a gap.
        """
        try:
            after = int(after_id)
        except (TypeError, ValueError):
            after = 0
        with self._lock:
            return [asdict(m) for m in self._messages if m.id > after]

    def snapshot(self, client_id=None, after_id=0, name=None):
        """One poll: presence refreshed, new messages, and the head count."""
        if client_id:
            self.see(client_id, name)
        return {
            "messages": self.since(after_id),
            "watching": self.watching(),
            "cursor": self._cursor(),
        }

    # -- internals ---------------------------------------------------------

    def _cursor(self):
        with self._lock:
            return self._next_id - 1

    def _append(self, kind, who, text, client_id=""):
        if not text:
            raise RoomError("empty message")
        msg = Message(
            id=self._next_id,
            kind=kind,
            who=who,
            text=text,
            t=self._clock(),
            client_id=client_id,
        )
        self._next_id += 1
        self._messages.append(msg)
        return asdict(msg)
