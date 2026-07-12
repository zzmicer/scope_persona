from scope.core.pipelines.lingbot_world.actions import parse_command
from scope.core.pipelines.lingbot_world.pipeline import LingbotWorldPipeline


def test_event_prompt_preserves_base_world_and_identity() -> None:
    pipeline = LingbotWorldPipeline.__new__(LingbotWorldPipeline)
    pipeline._prompt = "A woman in a black sweater sits beside a blue bed."
    pipeline._event_prompt = "rests her chin in both hands."

    composed = pipeline._composed_prompt()

    assert "woman in a black sweater" in composed
    assert "blue bed" in composed
    assert "rests her chin in both hands" in composed
    assert "Preserve her identity" in composed


def test_empty_event_prompt_returns_unchanged_base_prompt() -> None:
    pipeline = LingbotWorldPipeline.__new__(LingbotWorldPipeline)
    pipeline._prompt = "The persistent world"
    pipeline._event_prompt = ""

    assert pipeline._composed_prompt() == "The persistent world"


def test_beauty_script_commands_are_events_not_camera_motions() -> None:
    commands = [
        "event: gently runs one hand through her hair and looks toward the camera",
        "event: leans closer and rests her chin softly in both hands",
        "event: holds a small lit candle carefully in both hands",
    ]

    for command in commands:
        motions, event = parse_command(command)
        assert motions == []
        assert event
