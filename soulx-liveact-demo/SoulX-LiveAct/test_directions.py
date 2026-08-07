"""Parser tests for `directions.py` — no model, no GPU.

The cases below are the shapes actually observed coming out of Qwen2.5-1.5B on
the live demo, plus the format the new prompt asks for.
"""

import directions
from directions import Direction, parse, seconds_to_chunks


def test_the_motivating_example():
    d = parse("Aww, thank you! [she turns around slowly, 2s] Do you like it?")
    assert d.say == "Aww, thank you! Do you like it?"
    assert d.action == "She turns around slowly."
    assert d.hold == 1  # 2s = one chunk
    # asked for 2s, so she goes back afterwards rather than staying turned
    assert d.pose is None


def test_bracket_without_duration_falls_back_to_the_server_default():
    d = parse("Sure thing! [she waves at the camera]", default_hold=8)
    assert d.say == "Sure thing!"
    assert d.action == "She waves at the camera."
    assert d.hold == 8
    assert d.pose is None  # a wave is momentary


def test_posture_change_without_a_duration_becomes_the_resting_pose():
    d = parse("Okay, getting comfy. [she sits down on the floor]")
    assert d.action == "She sits down on the floor."
    assert d.pose == "She sits down on the floor."


def test_posture_phrasings_the_model_actually_uses_are_sustained():
    # Observed live on the pod: "sit down for a bit" came back as "takes a seat",
    # which the bare `sit` pattern missed and left as a momentary gesture.
    for action in (
        "[she takes a seat on the couch, relaxing]",
        "[she settles onto the cushion]",
        "[she curls up on the sofa]",
    ):
        assert parse(action).pose is not None, action


def test_posture_change_with_a_duration_stays_transient():
    d = parse("Just for a moment! [she sits down for 4 seconds]")
    assert d.pose is None
    assert d.hold == 2


def test_explicit_stays_makes_a_gesture_stick():
    d = parse("[she raises both arms and stays like that, 2s]")
    assert d.pose == "She raises both arms and stays like that."


def test_for_ten_seconds_rounds_up_to_whole_chunks():
    assert parse("[she waves for 3 seconds]").hold == 2
    assert parse("[she waves for 10s]").hold == 5
    assert parse("[she waves for 1 minute]").hold == 30


def test_duration_is_stripped_from_the_video_prompt():
    d = parse("[she nods at you for 2 sec]")
    assert d.action == "She nods at you."


def test_first_person_is_rewritten_to_third():
    assert parse("[I turn around]").action == "She turns around."
    assert parse("[I wave at you]").action == "She waves at you."
    assert parse("[I stretch my arms]").action == "She stretches her arms."


def test_asterisk_directions_are_understood():
    d = parse("Hehe *she covers her mouth and giggles* you're sweet.")
    assert d.say == "Hehe you're sweet."
    assert d.action == "She covers her mouth and giggles."


def test_single_emphasized_word_is_left_in_the_speech():
    d = parse("You're *so* kind!")
    assert d.action is None
    assert d.say == "You're *so* kind!"


def test_bare_third_person_sentence_is_promoted_not_spoken():
    # The measured 1.5B failure: motion narrated in the speech slot.
    d = parse("She waves at you with a bright smile. Thank you so much!", name="Chano")
    assert d.action == "She waves at you with a bright smile."
    assert d.say == "Thank you so much!"


def test_speech_that_merely_mentions_her_is_not_promoted():
    d = parse("She sounds like someone I would like.")
    assert d.action is None
    assert d.say == "She sounds like someone I would like."


def test_silent_action_leaves_nothing_to_speak():
    d = parse("[she waves]")
    assert d.say == ""
    assert d.action == "She waves."


def test_legacy_json_still_works():
    d = parse('{"say": "Hi there!", "action": "She waves.", "pose": null}')
    assert (d.say, d.action, d.pose) == ("Hi there!", "She waves.", None)


def test_concatenated_json_objects_are_merged_not_spoken():
    # This shape used to leak the whole raw string into `say` and get read aloud.
    raw = '{"say": "Hello!"}\n{"action": "She nods."}\n{"pose": null}'
    d = parse(raw)
    assert d.say == "Hello!"
    assert d.action == "She nods."


def test_json_with_the_motion_in_say_is_still_recovered():
    d = parse('{"say": "She waves at you, her hand high.", "action": null}')
    assert d.action == "She waves at you, her hand high."
    assert d.say == ""


def test_multiple_directions_merge_into_one_motion():
    d = parse("[she stands up] Look! [she points at the sky]")
    assert d.action == "She stands up. She points at the sky."
    assert d.say == "Look!"


def test_empty_and_junk_inputs_are_safe():
    assert parse("") == Direction(say="")
    assert parse(None) == Direction(say="")
    assert parse("   ").say == ""


def test_hold_is_capped():
    assert seconds_to_chunks(10_000) == directions.MAX_HOLD_CHUNKS
    assert seconds_to_chunks(0) is None
    assert seconds_to_chunks(None) is None


def test_script_round_trips_the_format_back_into_history():
    d = parse("Sure! [she turns around, 2s] Ta-da!")
    assert d.script == "Sure! Ta-da! [She turns around.]"
