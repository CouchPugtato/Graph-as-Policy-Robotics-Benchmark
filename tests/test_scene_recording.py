from pathlib import Path
from unittest.mock import Mock

from gap_isaac.scene_recording import SceneRecorder


def recorder_stub():
    recorder = SceneRecorder.__new__(SceneRecorder)
    recorder.steps = 0
    recorder.interval = 4
    recorder.fps = 30
    recorder.frames = 60
    recorder.output = Path("scene.mp4")
    recorder.closed = False
    recorder.encoder = Mock()
    recorder.camera = Mock()
    recorder.encode = Mock()
    return recorder


def test_rendering_is_sampled_without_extra_physics_steps():
    recorder = recorder_stub()
    for _ in range(3):
        assert not recorder.render_due()
        recorder.after_step()
    assert recorder.render_due()
    recorder.after_step()
    recorder.encode.assert_called_once()
    assert recorder.steps == 4


def test_finalization_is_idempotent_and_reports_duration():
    recorder = recorder_stub()
    assert recorder.close()["duration_s"] == 2
    recorder.close()
    recorder.encoder.finalize_encoding.assert_called_once()
