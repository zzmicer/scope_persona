import queue

from scope.server.pipeline_processor import PipelineProcessor


def _processor() -> PipelineProcessor:
    processor = PipelineProcessor.__new__(PipelineProcessor)
    processor.pipeline_id = "test"
    processor.parameters_queue = queue.Queue(maxsize=8)
    return processor


def test_parameter_updates_coalesce_to_latest_values() -> None:
    processor = _processor()

    processor.update_parameters({"prompt": "old", "move_speed": 0.1})
    processor.update_parameters({"prompt": "new"})

    assert processor.parameters_queue.qsize() == 1
    assert processor.parameters_queue.get_nowait() == {
        "prompt": "new",
        "move_speed": 0.1,
    }


def test_controller_updates_sum_mouse_and_keep_latest_buttons() -> None:
    processor = _processor()

    processor.update_parameters(
        {"ctrl_input": {"button": ["KeyW"], "mouse": [3.5, -2.0]}}
    )
    processor.update_parameters(
        {"ctrl_input": {"button": ["KeyD"], "mouse": [-1.0, 4.0]}}
    )

    assert processor.parameters_queue.qsize() == 1
    assert processor.parameters_queue.get_nowait() == {
        "ctrl_input": {"button": ["KeyD"], "mouse": [2.5, 2.0]}
    }
