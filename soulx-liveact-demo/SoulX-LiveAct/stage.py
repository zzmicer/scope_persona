"""Turn-taking for a single live session shared by many viewers.

One GPU renders one session, so exactly one visitor at a time may drive the
character. Everyone else watches the same stream. This module owns that
arrangement and nothing else: it hands out short turns, issues the token the
control endpoints authorize against, and meters the one action that costs real
money (the appearance change).

Deliberately stdlib-only and free of Flask/torch imports so the queue can be
tested on a laptop without the model.

Time is injected (`clock`) rather than read from `time.time()` directly, because
every interesting behaviour here is a deadline and tests should not sleep.
"""

import secrets
import threading
from collections import deque
from dataclasses import dataclass

DEFAULT_TURN_SECONDS = 90.0
# A waiter who closed the tab must not hold up the line. The client heartbeats
# on a timer; miss a few and the slot is forfeit. Generous enough to survive a
# backgrounded phone tab, short enough that the advertised ETA stays honest.
DEFAULT_IDLE_TIMEOUT = 30.0
# A promoted visitor who never says anything back was a closed tab that happened
# to still look live when the seat opened. Without this, that tab burns a whole
# turn of dead air in front of everyone; with it, dead air is bounded by one
# grace window. Must exceed the client heartbeat interval.
DEFAULT_NO_SHOW_GRACE = 10.0
DEFAULT_OUTFIT_PER_TURN = 1
MAX_NAME_LEN = 24


class StageError(Exception):
    """A request that the stage rules reject. `status` is the HTTP code to use."""

    status = 400

    def __init__(self, message, status=None):
        super().__init__(message)
        self.message = message
        if status is not None:
            self.status = status


class NotYourTurn(StageError):
    status = 403


class OutOfBudget(StageError):
    status = 429


@dataclass
class Turn:
    """The active driver's turn. `token` is the bearer of control authority."""

    client_id: str
    name: str
    token: str
    started: float
    expires: float
    outfit_left: int
    # Flips on the first sign of life from the driver (a heartbeat or any
    # authorized control call). See DEFAULT_NO_SHOW_GRACE.
    confirmed: bool = False

    def remaining(self, now):
        return max(0.0, self.expires - now)


@dataclass
class _Waiter:
    client_id: str
    name: str
    joined: float
    seen: float
    # Turns taken this visit — surfaced so the UI can say "you're up again" and
    # so a future fairness policy has the counter it needs without a migration.
    turns: int = 0


@dataclass
class StageConfig:
    turn_seconds: float = DEFAULT_TURN_SECONDS
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT
    no_show_grace: float = DEFAULT_NO_SHOW_GRACE
    outfit_per_turn: int = DEFAULT_OUTFIT_PER_TURN
    # Global ceiling on appearance edits, since each one is a paid API call and
    # a session restart. Refilled by the caller (see `set_outfit_budget`).
    outfit_budget: int = 40


def sanitize_name(raw):
    """A display name safe to put on the overlay in front of everyone."""
    name = " ".join(str(raw or "").split())[:MAX_NAME_LEN]
    # Control characters would land in JSON broadcast to every viewer.
    name = "".join(ch for ch in name if ch.isprintable())
    return name or "guest"


class Stage:
    """Queue of visitors waiting to drive one live session.

    All public methods advance the schedule first, so callers never have to run
    a ticker thread: the queue moves whenever anyone looks at it. A stage no one
    is watching does not need to move.
    """

    def __init__(self, config=None, clock=None, on_turn_start=None, on_turn_end=None):
        self.config = config or StageConfig()
        self._clock = clock or __import__("time").time
        self._lock = threading.RLock()
        self._waiting = deque()  # _Waiter, front = next up
        self._by_id = {}  # client_id -> _Waiter, for O(1) heartbeat/leave
        self._current = None  # Turn | None
        self._outfit_spent = 0
        self._on_turn_start = on_turn_start
        self._on_turn_end = on_turn_end
        # Set when the stage is closed to new turns (moderation, shutdown).
        self._closed_reason = None

    # -- membership ---------------------------------------------------------

    def join(self, client_id, name=""):
        """Add a visitor to the line (idempotent), returning their view of it."""
        if not client_id:
            raise StageError("client_id required")
        with self._lock:
            self._advance()
            display = sanitize_name(name)
            waiter = self._by_id.get(client_id)
            if waiter is not None:
                waiter.seen = self._clock()
                waiter.name = display
            elif self._current is not None and self._current.client_id == client_id:
                # Already driving; renaming mid-turn is allowed, re-queueing is
                # not — otherwise a driver could hold the seat back-to-back.
                # Re-joining is also how a driver who reloaded proves presence.
                self._current.name = display
                self._current.confirmed = True
            else:
                waiter = _Waiter(
                    client_id=client_id,
                    name=display,
                    joined=self._clock(),
                    seen=self._clock(),
                )
                self._waiting.append(waiter)
                self._by_id[client_id] = waiter
            self._advance()
            return self._snapshot(client_id)

    def heartbeat(self, client_id):
        """Refresh a waiter's liveness. Cheap; called on a client timer."""
        with self._lock:
            waiter = self._by_id.get(client_id)
            if waiter is not None:
                waiter.seen = self._clock()
            elif self._current is not None and self._current.client_id == client_id:
                self._current.confirmed = True
            self._advance()
            return self._snapshot(client_id)

    def leave(self, client_id):
        with self._lock:
            waiter = self._by_id.pop(client_id, None)
            if waiter is not None:
                try:
                    self._waiting.remove(waiter)
                except ValueError:
                    pass
            if self._current is not None and self._current.client_id == client_id:
                self._end_turn("left")
            self._advance()

    # -- driving -----------------------------------------------------------

    def authorize(self, token):
        """Return the live Turn for `token`, or raise. The gate every control
        endpoint calls before touching the session."""
        with self._lock:
            self._advance()
            current = self._current
            if current is None:
                raise NotYourTurn("nobody is driving right now")
            if not token or not secrets.compare_digest(str(token), current.token):
                raise NotYourTurn("it is not your turn")
            # Driving something is the strongest possible proof of presence.
            current.confirmed = True
            return current

    def spend_outfit(self, token):
        """Consume one appearance-change credit. Raises if none is left, so the
        caller never starts a paid edit it cannot account for."""
        with self._lock:
            turn = self.authorize(token)
            if turn.outfit_left <= 0:
                raise OutOfBudget("you have already changed her outfit this turn")
            if self._outfit_spent >= self.config.outfit_budget:
                raise OutOfBudget("outfit changes are used up for now")
            turn.outfit_left -= 1
            self._outfit_spent += 1
            return turn.outfit_left

    def skip(self, token):
        """The driver ends their own turn early."""
        with self._lock:
            self.authorize(token)
            self._end_turn("skipped")
            self._advance()

    # -- moderation --------------------------------------------------------

    def force_end(self, reason="moderated"):
        with self._lock:
            if self._current is not None:
                self._end_turn(reason)
            self._advance()

    def close(self, reason="closed"):
        """Stop granting turns; watchers keep watching."""
        with self._lock:
            self._closed_reason = reason
            if self._current is not None:
                self._end_turn(reason)

    def open(self):
        with self._lock:
            self._closed_reason = None
            self._advance()

    def set_outfit_budget(self, total):
        with self._lock:
            self.config.outfit_budget = max(0, int(total))

    # -- views -------------------------------------------------------------

    def state(self, client_id=None):
        with self._lock:
            self._advance()
            return self._snapshot(client_id)

    # -- internals ---------------------------------------------------------

    def _advance(self):
        """Expire the current turn and ghosts, then promote the next waiter."""
        now = self._clock()
        if self._current is not None and now >= self._current.expires:
            self._end_turn("expired")
        if (
            self._current is not None
            and not self._current.confirmed
            and (now - self._current.started) > self.config.no_show_grace
        ):
            self._end_turn("no_show")
        # Drop the disappeared before promoting, so the seat never goes to a
        # closed tab while a live visitor waits behind it. Only the front is
        # checked (O(1) per look): a ghost further back costs an overstated ETA
        # until it reaches the front, not a stalled queue.
        timeout = self.config.idle_timeout
        while self._waiting and (now - self._waiting[0].seen) > timeout:
            ghost = self._waiting.popleft()
            self._by_id.pop(ghost.client_id, None)
        if self._current is None and self._waiting and self._closed_reason is None:
            waiter = self._waiting.popleft()
            self._by_id.pop(waiter.client_id, None)
            waiter.turns += 1
            self._current = Turn(
                client_id=waiter.client_id,
                name=waiter.name,
                token=secrets.token_urlsafe(16),
                started=now,
                expires=now + self.config.turn_seconds,
                outfit_left=self.config.outfit_per_turn,
            )
            if self._on_turn_start is not None:
                self._on_turn_start(self._current)

    def _end_turn(self, reason):
        turn = self._current
        self._current = None
        if turn is not None and self._on_turn_end is not None:
            # The callback clears the sustained pose so the next driver starts
            # from neutral rather than inheriting a stranger's held state.
            self._on_turn_end(turn, reason)

    def _snapshot(self, client_id):
        now = self._clock()
        current = self._current
        me = {"role": "watching", "position": None, "eta": None}
        if current is not None and current.client_id == client_id:
            me = {
                "role": "driving",
                "position": 0,
                "eta": 0.0,
                "token": current.token,
                "remaining": round(current.remaining(now), 1),
                "outfit_left": current.outfit_left,
            }
        elif client_id is not None:
            for index, waiter in enumerate(self._waiting):
                if waiter.client_id == client_id:
                    ahead = current.remaining(now) if current is not None else 0.0
                    me = {
                        "role": "queued",
                        "position": index + 1,
                        "eta": round(ahead + index * self.config.turn_seconds, 1),
                    }
                    break
        return {
            "driver": None
            if current is None
            else {
                "name": current.name,
                "remaining": round(current.remaining(now), 1),
                "turn_seconds": self.config.turn_seconds,
            },
            "waiting": len(self._waiting),
            "closed": self._closed_reason,
            "outfit_left_global": max(
                0, self.config.outfit_budget - self._outfit_spent
            ),
            "me": me,
        }
