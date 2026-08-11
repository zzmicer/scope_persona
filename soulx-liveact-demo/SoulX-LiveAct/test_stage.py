import pytest
from stage import (
    NotYourTurn,
    OutOfBudget,
    Stage,
    StageConfig,
    StageError,
    sanitize_name,
)


class FakeClock:
    """Deadlines are the whole subject here, so time is driven, never slept."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_stage(**kwargs):
    clock = FakeClock()
    config = StageConfig(
        turn_seconds=kwargs.pop("turn_seconds", 90.0),
        idle_timeout=kwargs.pop("idle_timeout", 30.0),
        # Off unless a test is about it, so no other test has to keep the
        # driver alive to say something about the queue.
        no_show_grace=kwargs.pop("no_show_grace", 1e9),
        outfit_per_turn=kwargs.pop("outfit_per_turn", 1),
        outfit_budget=kwargs.pop("outfit_budget", 40),
    )
    return Stage(config=config, clock=clock, **kwargs), clock


def test_first_visitor_starts_driving_immediately():
    stage, _ = make_stage()
    state = stage.join("a", "Alex")
    assert state["me"]["role"] == "driving"
    assert state["driver"]["name"] == "Alex"
    assert state["me"]["token"]
    assert state["waiting"] == 0


def test_second_visitor_queues_with_position_and_eta():
    stage, clock = make_stage(turn_seconds=90.0)
    stage.join("a", "Alex")
    clock.advance(30)
    state = stage.join("b", "Bo")
    assert state["me"]["role"] == "queued"
    assert state["me"]["position"] == 1
    # 60s left of Alex's turn, and nobody ahead of Bo.
    assert state["me"]["eta"] == pytest.approx(60.0)
    third = stage.join("c", "Cy")
    assert third["me"]["position"] == 2
    assert third["me"]["eta"] == pytest.approx(150.0)


def beat(stage, clock, seconds, *client_ids):
    """Advance time the way live clients do: heartbeating every 10s."""
    remaining = seconds
    while remaining > 0:
        step = min(10.0, remaining)
        clock.advance(step)
        remaining -= step
        for client_id in client_ids:
            stage.heartbeat(client_id)


def test_turn_expires_and_next_waiter_is_promoted():
    stage, clock = make_stage(turn_seconds=90.0)
    first = stage.join("a", "Alex")["me"]["token"]
    stage.join("b", "Bo")
    beat(stage, clock, 90.0, "a", "b")
    state = stage.state("b")
    assert state["driver"]["name"] == "Bo"
    assert state["me"]["role"] == "driving"
    # Alex's token dies with Alex's turn.
    with pytest.raises(NotYourTurn):
        stage.authorize(first)


def test_authorize_rejects_wrong_and_missing_tokens():
    stage, _ = make_stage()
    token = stage.join("a", "Alex")["me"]["token"]
    assert stage.authorize(token).client_id == "a"
    for bad in (None, "", "nope"):
        with pytest.raises(NotYourTurn):
            stage.authorize(bad)


def test_authorize_raises_when_nobody_is_driving():
    stage, _ = make_stage()
    with pytest.raises(NotYourTurn):
        stage.authorize("anything")


def test_stale_waiter_is_evicted_before_being_promoted():
    """A closed tab must not take the seat while a live visitor waits."""
    stage, clock = make_stage(turn_seconds=100.0, idle_timeout=30.0)
    stage.join("a", "Alex")
    stage.join("ghost", "Ghost")
    stage.join("c", "Cy")
    beat(stage, clock, 100.0, "a", "c")  # Ghost never heartbeats
    state = stage.state("c")
    assert state["driver"]["name"] == "Cy"
    assert state["waiting"] == 0


def test_ghost_that_wins_the_seat_forfeits_after_the_grace_window():
    """The seat can open while a dead tab still looks live. Dead air must be
    bounded by the grace window, not by a whole turn."""
    stage, clock = make_stage(turn_seconds=90.0, idle_timeout=30.0, no_show_grace=10.0)
    stage.join("a", "Alex")
    stage.join("ghost", "Ghost")
    stage.join("c", "Cy")
    stage.skip(stage.state("a")["me"]["token"])
    assert stage.state("c")["driver"]["name"] == "Ghost"  # looked live, was not
    clock.advance(10.1)
    stage.heartbeat("c")
    state = stage.state("c")
    assert state["driver"]["name"] == "Cy"
    assert state["me"]["role"] == "driving"


def test_a_driver_who_answers_keeps_the_full_turn():
    stage, clock = make_stage(turn_seconds=90.0, no_show_grace=10.0)
    stage.join("a", "Alex")
    stage.join("b", "Bo")
    beat(stage, clock, 30.0, "a", "b")
    state = stage.state("a")
    assert state["driver"]["name"] == "Alex"
    assert state["me"]["remaining"] == pytest.approx(60.0)


def test_a_control_call_alone_proves_presence():
    """A driver typing an action but never heartbeating is present."""
    stage, clock = make_stage(turn_seconds=90.0, no_show_grace=10.0)
    token = stage.join("a", "Alex")["me"]["token"]
    clock.advance(9.0)
    stage.authorize(token)
    clock.advance(9.0)
    assert stage.state("a")["driver"]["name"] == "Alex"


def test_heartbeat_keeps_a_waiter_in_line():
    stage, clock = make_stage(turn_seconds=100.0, idle_timeout=30.0)
    stage.join("a", "Alex")
    stage.join("b", "Bo")
    for _ in range(4):
        clock.advance(20)
        stage.heartbeat("b")
    state = stage.state("b")
    assert state["me"]["role"] == "queued"
    assert state["me"]["position"] == 1


def test_leaving_mid_turn_hands_the_seat_over():
    stage, _ = make_stage()
    stage.join("a", "Alex")
    stage.join("b", "Bo")
    stage.leave("a")
    assert stage.state("b")["me"]["role"] == "driving"


def test_leaving_the_queue_shortens_the_line():
    stage, _ = make_stage()
    stage.join("a", "Alex")
    stage.join("b", "Bo")
    stage.join("c", "Cy")
    stage.leave("b")
    assert stage.state("c")["me"]["position"] == 1


def test_skip_ends_your_own_turn_early():
    stage, _ = make_stage()
    token = stage.join("a", "Alex")["me"]["token"]
    stage.join("b", "Bo")
    stage.skip(token)
    assert stage.state("b")["driver"]["name"] == "Bo"


def test_rejoining_while_driving_does_not_requeue_you():
    """Otherwise a driver could keep the seat back to back."""
    stage, _ = make_stage()
    stage.join("a", "Alex")
    state = stage.join("a", "Alex Renamed")
    assert state["me"]["role"] == "driving"
    assert state["waiting"] == 0
    assert state["driver"]["name"] == "Alex Renamed"


def test_rejoining_while_queued_is_idempotent():
    stage, _ = make_stage()
    stage.join("a", "Alex")
    stage.join("b", "Bo")
    again = stage.join("b", "Bo")
    assert again["me"]["position"] == 1
    assert again["waiting"] == 1


def test_outfit_credit_is_one_per_turn():
    stage, _ = make_stage(outfit_per_turn=1)
    token = stage.join("a", "Alex")["me"]["token"]
    assert stage.spend_outfit(token) == 0
    with pytest.raises(OutOfBudget):
        stage.spend_outfit(token)


def test_outfit_credit_refreshes_on_the_next_turn():
    stage, clock = make_stage(turn_seconds=10.0, outfit_per_turn=1)
    first = stage.join("a", "Alex")["me"]["token"]
    stage.join("b", "Bo")
    stage.spend_outfit(first)
    clock.advance(10)
    second = stage.state("b")["me"]["token"]
    assert stage.spend_outfit(second) == 0


def test_global_outfit_budget_stops_paid_edits():
    stage, clock = make_stage(turn_seconds=10.0, outfit_per_turn=1, outfit_budget=1)
    first = stage.join("a", "Alex")["me"]["token"]
    stage.join("b", "Bo")
    stage.spend_outfit(first)
    clock.advance(10)
    second = stage.state("b")["me"]["token"]
    with pytest.raises(OutOfBudget):
        stage.spend_outfit(second)
    assert stage.state("b")["outfit_left_global"] == 0


def test_turn_callbacks_fire_with_a_reason():
    started, ended = [], []
    clock = FakeClock()
    stage = Stage(
        config=StageConfig(turn_seconds=10.0),
        clock=clock,
        on_turn_start=lambda turn: started.append(turn.name),
        on_turn_end=lambda turn, reason: ended.append((turn.name, reason)),
    )
    stage.join("a", "Alex")
    stage.join("b", "Bo")
    clock.advance(10)
    stage.state()
    assert started == ["Alex", "Bo"]
    assert ended == [("Alex", "expired")]
    stage.force_end()
    assert ended[-1] == ("Bo", "moderated")


def test_closing_the_stage_stops_granting_turns():
    stage, _ = make_stage()
    stage.join("a", "Alex")
    stage.close("maintenance")
    state = stage.join("b", "Bo")
    assert state["driver"] is None
    assert state["closed"] == "maintenance"
    assert state["me"]["role"] == "queued"
    stage.open()
    assert stage.state("b")["me"]["role"] == "driving"


def test_join_requires_a_client_id():
    stage, _ = make_stage()
    with pytest.raises(StageError):
        stage.join("", "Alex")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", "guest"),
        (None, "guest"),
        ("   ", "guest"),
        ("Alex  Kim", "Alex Kim"),
        ("bad\x00name", "badname"),
        ("x" * 50, "x" * 24),
        ("line\nbreak", "line break"),
    ],
)
def test_sanitize_name(raw, expected):
    assert sanitize_name(raw) == expected
