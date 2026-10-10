import importlib.util
from pathlib import Path

import pytest

from gap_isaac.contact_telemetry import scale_contact_frame_count


@pytest.mark.parametrize("frames", [2, 3])
@pytest.mark.parametrize("hz", [60, 120])
def test_contact_debounce_duration_is_preserved(frames, hz) -> None:
    assert scale_contact_frame_count(frames, hz) / hz == pytest.approx(frames / 60)


@pytest.mark.parametrize("frames,hz", [(0, 120), (3, 240), (3, 0)])
def test_contact_debounce_rejects_unbounded_rates(frames, hz) -> None:
    with pytest.raises(ValueError, match="60 or 120"):
        scale_contact_frame_count(frames, hz)


def load_runner():
    path = Path(__file__).resolve().parents[1] / "isaac/validate_graph_pick_place.py"
    spec = importlib.util.spec_from_file_location("graph_runner_options", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("hz", [60, 120])
def test_runner_accepts_bounded_physics_rate_without_loading_isaac(monkeypatch, hz) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--stage",
            "scene.usd",
            "--task",
            "task.yaml",
            "--graph",
            "graph.json",
            "--physics-hz",
            str(hz),
        ],
    )
    assert load_runner().parse_args().physics_hz == hz


def test_runner_rejects_higher_resource_physics_rate(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--stage",
            "scene.usd",
            "--task",
            "task.yaml",
            "--graph",
            "graph.json",
            "--physics-hz",
            "240",
        ],
    )
    with pytest.raises(SystemExit):
        load_runner().parse_args()
