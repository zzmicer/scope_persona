import pytest
from room import Room, RoomError, TooFast


class FakeClock:
    """Presence and flood control are both deadlines, so time is driven."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_room(**kwargs):
    clock = FakeClock()
    return Room(clock=clock, **kwargs), clock


# -- talking ----------------------------------------------------------------


def test_anyone_may_post_without_holding_the_seat():
    room, _ = make_room()
    msg = room.post("watcher", "wave at us!", name="anna")
    assert msg["kind"] == "chat"
    assert msg["who"] == "anna"
    assert msg["text"] == "wave at us!"


def test_message_ids_increase_so_clients_can_ask_for_the_tail():
    room, _ = make_room()
    first = room.post("a", "one", name="anna")
    second = room.post("b", "two", name="bob")
    assert second["id"] > first["id"]
    assert [m["text"] for m in room.since(first["id"])] == ["two"]
    assert room.since(second["id"]) == []


def test_since_zero_returns_the_whole_buffer():
    room, _ = make_room()
    room.post("a", "one", name="anna")
    room.post("b", "two", name="bob")
    assert len(room.since(0)) == 2


def test_a_chat_message_carries_its_author_so_the_driver_can_replay_it():
    """No separate suggestion type: the driver taps a message to act on it."""
    room, _ = make_room()
    msg = room.post("watcher", "spin around", name="anna")
    assert msg["client_id"] == "watcher"


def test_an_already_authorized_line_is_not_charged_to_the_flood_allowance():
    """The driver paid at the turn gate; a long turn must not throttle them out."""
    room, _ = make_room(burst=1, min_interval=1.0)
    for _ in range(5):
        assert room.said("anna", "wave", client_id="a")["who"] == "anna"


def test_an_echoed_line_still_reads_as_chat_and_keeps_its_author():
    room, _ = make_room()
    msg = room.said("anna", "spin around", client_id="a")
    assert msg["kind"] == "chat"
    assert msg["client_id"] == "a"


def test_persona_and_system_lines_carry_no_client_id():
    room, _ = make_room()
    assert room.persona("hi!")["client_id"] == ""
    assert room.system("anna is driving")["client_id"] == ""
    assert room.persona("hi!")["kind"] == "persona"
    assert room.system("x")["kind"] == "system"


def test_history_is_bounded_and_keeps_the_newest():
    room, _ = make_room(history=3)
    for i in range(6):
        room.system(f"m{i}")
    kept = [m["text"] for m in room.since(0)]
    assert kept == ["m3", "m4", "m5"]


def test_a_client_further_behind_than_the_buffer_gets_what_is_left():
    """A conversation, not a ledger -- a gap beats replaying an hour of backlog."""
    room, _ = make_room(history=2)
    stale = room.system("first")["id"]
    for i in range(5):
        room.system(f"m{i}")
    tail = room.since(stale)
    assert [m["text"] for m in tail] == ["m3", "m4"]


# -- hygiene ----------------------------------------------------------------


def test_empty_and_whitespace_messages_are_rejected():
    room, _ = make_room()
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(RoomError):
            room.post("a", bad, name="anna")


def test_text_is_capped_and_collapsed():
    room, _ = make_room()
    msg = room.post("a", "x" * 500, name="anna")
    assert len(msg["text"]) == 240
    assert room.post("b", "too   many\n\nspaces", name="bob")["text"] == (
        "too many spaces"
    )


def test_control_characters_never_reach_the_room():
    room, _ = make_room()
    msg = room.post("a", "hi\x00\x07there", name="anna")
    assert msg["text"] == "hithere"


def test_names_are_sanitized_on_the_way_in():
    room, _ = make_room()
    assert room.post("a", "hi", name="  ")["who"] == "guest"
    assert len(room.post("b", "hi", name="n" * 99)["who"]) == 24


def test_posting_requires_a_client_id():
    room, _ = make_room()
    with pytest.raises(RoomError):
        room.post("", "hi", name="anna")


# -- flood control ----------------------------------------------------------


def test_a_burst_is_allowed_then_throttled():
    room, clock = make_room(burst=3, min_interval=1.0)
    for i in range(3):
        room.post("a", f"m{i}", name="anna")
        clock.advance(0.01)
    with pytest.raises(TooFast):
        room.post("a", "flood", name="anna")


def test_allowance_accrues_while_a_client_is_quiet():
    room, clock = make_room(burst=2, min_interval=1.0)
    room.post("a", "one", name="anna")
    room.post("a", "two", name="anna")
    with pytest.raises(TooFast):
        room.post("a", "three", name="anna")
    clock.advance(1.0)
    assert room.post("a", "three", name="anna")["text"] == "three"


def test_one_flooder_does_not_throttle_anyone_else():
    room, _ = make_room(burst=1, min_interval=1.0)
    room.post("flooder", "spam", name="mallory")
    with pytest.raises(TooFast):
        room.post("flooder", "spam", name="mallory")
    assert room.post("bystander", "hello", name="bob")["text"] == "hello"


# -- presence ---------------------------------------------------------------


def test_watching_counts_everyone_recently_seen():
    room, _ = make_room()
    room.see("a", "anna")
    room.see("b", "bob")
    assert room.watching() == 2


def test_a_client_that_stopped_polling_is_swept():
    room, clock = make_room(presence_timeout=45.0)
    room.see("a", "anna")
    room.see("b", "bob")
    clock.advance(46.0)
    room.see("a", "anna")  # only anna is still polling
    assert room.watching() == 1


def test_leaving_removes_you_immediately():
    room, _ = make_room()
    room.see("a", "anna")
    room.see("b", "bob")
    room.leave("a")
    assert room.watching() == 1


def test_posting_counts_as_presence():
    room, _ = make_room()
    room.post("a", "hi", name="anna")
    assert room.watching() == 1


def test_a_rename_follows_the_client():
    room, _ = make_room()
    room.see("a", "anna")
    assert room.post("a", "hi", name="annabel")["who"] == "annabel"
    assert room.name_of("a") == "annabel"


def test_polling_without_a_name_keeps_the_one_already_known():
    room, _ = make_room()
    room.see("a", "anna")
    room.see("a")
    assert room.name_of("a") == "anna"


# -- the single poll --------------------------------------------------------


def test_snapshot_refreshes_presence_and_returns_only_new_messages():
    room, _ = make_room()
    first = room.post("a", "one", name="anna")
    snap = room.snapshot(client_id="b", after_id=first["id"], name="bob")
    assert snap["messages"] == []
    assert snap["watching"] == 2  # the poll itself put bob in the room
    room.post("a", "two", name="anna")
    assert [m["text"] for m in room.snapshot("b", snap["cursor"])["messages"]] == [
        "two"
    ]


def test_cursor_lets_a_fresh_client_start_at_the_end():
    room, _ = make_room()
    room.post("a", "old news", name="anna")
    snap = room.snapshot(client_id="b", after_id=0, name="bob")
    assert len(snap["messages"]) == 1
    assert room.snapshot("b", snap["cursor"])["messages"] == []
