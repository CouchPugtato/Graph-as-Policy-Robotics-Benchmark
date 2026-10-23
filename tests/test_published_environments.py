import json
from types import SimpleNamespace

import numpy as np
import pytest

from gap_harness.models import Task
from gap_harness.preflight import check_runner
from gap_libero.benchmarks import vab


@pytest.fixture
def published_definition(tmp_path, monkeypatch):
    task = {
        "id": "test.published",
        "language": "place the target into the destination",
        "inits": [{}] * 50,
        "success": {"predicate": "on_top_of", "args": {}},
        "horizon": 800,
    }
    path = tmp_path / "original.yaml"
    path.write_text("original task bytes", encoding="utf-8")
    monkeypatch.setattr(vab, "ROOT", tmp_path)
    monkeypatch.setattr(vab, "definition", lambda case: (path, task))
    return path, task


@pytest.mark.parametrize("case", vab.CASES)
def test_published_manifests_reuse_shared_actions_and_remain_separate(published_definition, case):
    path = vab.task_manifest(case)
    task = Task.model_validate_json(path.read_text(encoding="utf-8"))
    assert task.task_id == f"gap-published-{case}-local-qwen"
    assert task.seeds == list(range(10))
    assert "published" in task.runner.command
    assert task.runner.timeout_s == 300
    assert "scenario" not in task.runner.command
    actions = {a.name for skill in task.skills for a in skill.actions}
    assert "identify_target" in actions and "check_placement" in actions
    assert "turn_stove_on" not in actions
    assert ("select_next_item" in actions) == (case == "packing")
    assert "evaluation.task_success" in task.required_true
    assert published_definition[0].read_text() == "original task bytes"


def test_generation_is_blocked_without_a_passing_binding_probe(published_definition, monkeypatch):
    path = vab.task_manifest("popcorn")
    harness = SimpleNamespace(
        task=Task.model_validate_json(path.read_text()),
        task_path=path,
        progress=SimpleNamespace(event=lambda text: None),
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=1, stdout=json.dumps({"ok": False, "detail": "not ready"})
        )

    monkeypatch.setattr("gap_harness.preflight.subprocess.run", run)
    with pytest.raises(RuntimeError, match="not ready"):
        check_runner(harness)
    assert calls[0][-2:] == ["--published-case", "popcorn"]
    assert "--enable-motion" not in calls[0]


def test_readiness_is_invalidated_when_bindings_change(published_definition, monkeypatch):
    monkeypatch.setattr(vab, "binding_signature", lambda case: "current")
    with pytest.raises(RuntimeError, match="smoke test"):
        vab.readiness("packing")
    marker = vab.ROOT / "cache/vab-readiness/packing.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"ok": True, "binding_signature": "current"}))
    assert vab.readiness("packing")["ok"]
    monkeypatch.setattr(vab, "binding_signature", lambda case: "changed")
    with pytest.raises(RuntimeError, match="smoke test"):
        vab.readiness("packing")


def test_geometry_never_falls_back_to_object_pose():
    class Model:
        top_offset = [0, 0, 0.04]
        bottom_offset = [0, 0, -0.04]

    grid = np.array(
        [[x, y, 0.08] for x in np.linspace(0.1, 0.15, 10) for y in np.linspace(0.2, 0.25, 10)]
    )
    result = vab.geometry(grid, "alphabet_soup", objects={"alphabet_soup": Model()})
    assert result["position_m"] == pytest.approx([0.125, 0.225, 0.04])
    assert result["uses_evaluator_pose"] is False
    with pytest.raises(ValueError, match="visible RGB-D"):
        vab.geometry(np.empty((0, 3)), "alphabet_soup", objects={"alphabet_soup": Model()})


@pytest.mark.parametrize("angle", [0, 0.7, 1.6])
def test_pan_grasp_ignores_a_touching_stove_region(angle):
    pytest.importorskip("scipy")
    grid = np.linspace(-0.095, 0.095, 80)
    bowl = np.array([[x, y, 0.91] for x in grid for y in grid if x * x + y * y < 0.09**2])
    theta = np.linspace(-np.pi, np.pi, 500)
    rim = np.column_stack((0.09 * np.cos(theta), 0.09 * np.sin(theta), np.full(500, 0.922)))
    handle = np.array(
        [[x, y, 0.95] for x in np.linspace(0.12, 0.25, 120) for y in np.linspace(-0.011, 0.011, 14)]
    )
    stove = np.array(
        [[x, y, 0.96] for x in np.linspace(-0.09, 0.09, 50) for y in np.linspace(0.1, 0.3, 50)]
    )
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    cloud = np.vstack((bowl, rim, handle, stove))
    cloud[:, :2] = cloud[:, :2] @ rotation.T + [0.1, -0.05]
    result = vab.geometry(cloud, "chefmate_8_frypan", floor=0.9)
    expected = np.array([0.17, 0]) @ rotation.T + [0.1, -0.05]
    assert result["position_m"][:2] == pytest.approx(expected, abs=0.01)
    assert 0.94 < result["position_m"][2] < 0.96
    assert result["uses_evaluator_pose"] is False


def test_kitchen_candidates_preserve_pan_and_add_separate_stove(monkeypatch):
    full = np.ones((4, 4), dtype=bool)
    small = full.copy()
    small[:2] = False

    def candidates(bundle, **kwargs):
        mask = small if kwargs.get("color_threshold") else full
        return [{"mask": mask, "pixels": 5000, "box_xyxy": [0, 0, 4, 4], "id": "C1"}]

    monkeypatch.setattr("gap_libero.vision.native_candidates", candidates)
    result = vab.published_candidates(None, floor=0.9, kitchen=True)
    assert [r["id"] for r in result] == ["C1", "C2"]
    assert np.array_equal(result[0]["mask"], full)
    assert np.array_equal(result[1]["mask"], small)


def test_stove_geometry_uses_its_plane_not_a_touching_pan():
    stove = np.array(
        [[x, y, 0.96] for x in np.linspace(-0.09, 0.09, 80) for y in np.linspace(0.1, 0.3, 80)]
    )
    pan = np.array(
        [[x, y, 0.92] for x in np.linspace(-0.1, 0.1, 70) for y in np.linspace(-0.1, 0.1, 70)]
    )
    handle = np.array(
        [[x, y, 0.95] for x in np.linspace(-0.012, 0.012, 10) for y in np.linspace(0.12, 0.25, 50)]
    )
    result = vab.geometry(np.vstack((stove, pan, handle)), "flat_stove", floor=0.9)
    assert result["position_m"][:2] == pytest.approx([-0.03, 0.2], abs=0.003)
    assert result["uses_evaluator_pose"] is False


@pytest.mark.parametrize("axis", [-np.pi - 0.02, -np.pi + 0.02, np.pi - 0.02, np.pi + 0.02])
def test_thin_packages_do_not_flip_the_wrist_at_the_angle_boundary(axis):
    for current in (0, 1.57, -1.57):
        assert 1.5 < vab.jaw_yaw("cream_cheese", axis, current, thin=True) < 1.65
