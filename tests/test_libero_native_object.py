from types import SimpleNamespace

import numpy as np
import pytest

from gap_harness.engine import Harness
from gap_libero.grocery_comparison import build_action_plan
from gap_libero.vision import native_candidates, native_geometry


def test_depth_edges_separate_touching_image_regions(monkeypatch):
    from gap_libero import vision

    y, x = np.indices((8, 12))
    points = np.stack((x * 0.007 + (x >= 6) * 0.15, y * 0.007,
                       0.02 + y * 0.008), axis=-1)
    bundle = SimpleNamespace(to_gap_camera=lambda: {"depth": np.ones((8, 12))})
    monkeypatch.setattr(vision, "masked_world_points", lambda b, m: points[m])
    candidates = native_candidates(bundle)
    assert len(candidates) == 2
    assert all(candidate["pixels"] == 48 for candidate in candidates)
    assert not np.any(candidates[0]["mask"] & candidates[1]["mask"])


def test_native_candidate_foreground_is_bounded(monkeypatch):
    from gap_libero import vision

    bundle = SimpleNamespace(to_gap_camera=lambda: {"depth": np.ones((240, 240))})
    monkeypatch.setattr(vision, "masked_world_points",
                        lambda b, m: np.tile([0.1, 0.1, 0.05], (int(m.sum()), 1)))
    with pytest.raises(ValueError, match="pixel budget"):
        native_candidates(bundle)


def test_native_can_geometry_uses_surface_points_only():
    x, y = np.meshgrid(np.linspace(-0.03, 0.03, 8), np.linspace(-0.03, 0.03, 8))
    cloud = np.column_stack((x.ravel() + 0.1, y.ravel() - 0.2, np.full(64, 0.08)))
    geometry = native_geometry(cloud, "alphabet_soup")
    assert np.allclose(geometry["position_m"], [0.1, -0.2, 0.04])
    assert geometry["uses_evaluator_pose"] is False
    with pytest.raises(ValueError, match="supported upright"):
        native_geometry(cloud + [0, 0, 0.2], "alphabet_soup")


def test_native_plan_never_falls_back_to_object_poses():
    with pytest.raises(ValueError, match="image derived"):
        build_action_plan(object(), 0, native=True)


def test_native_plan_keeps_fingers_above_basket_rim():
    backend = SimpleNamespace(workspace=np.array([[-0.8, -0.8, 0.015], [0.8, 0.8, 1.2]]),
                              max_steps=1000)
    geometry = {"alphabet_soup": {"position_m": [-0.12, -0.24, 0.04]},
                "basket": {"position_m": [0.01, 0.25, 0], "yaw_rad": 0}}
    plan = build_action_plan(backend, 3, geometry, native=True)
    assert plan.suite == "libero_object"
    assert plan.init_state_id == 3
    assert plan.actions["position_for_release"].position_m[2] == 0.24
    assert plan.actions["lift_grasped"].require_grasp


def test_aliases_do_not_change_native_objects_or_goal():
    from test_libero_backend import make_backend

    backend, env = make_backend(object_aliases={"alphabet_soup": "target"})
    assert backend.target_object("alphabet_soup") is env.env.objects_dict["target"]
    assert "alphabet_soup" not in env.env.objects_dict
    assert backend.task_success() is False


def test_post_goal_verification_does_not_override_horizon_or_failures():
    from test_libero_backend import make_backend

    backend, env = make_backend(verify_after_goal=True, motion_enabled=True)
    env.success, env.force_done = True, True
    backend.step([0] * 7)
    assert backend.done is False
    env.env.done = True
    backend.step([0] * 7)
    assert backend.done is True
    backend, env = make_backend(verify_after_goal=True, motion_enabled=True)
    env.success, env.force_done = False, True
    backend.step([0] * 7)
    assert backend.done is True


def test_native_task_manifest_selects_ten_saved_states(tmp_path):
    from gap_harness.cli import ROOT

    task = Harness(ROOT / "harness/tasks/libero-native-object-local-qwen.json", tmp_path).task
    assert task.seeds == list(range(10))
    assert "--native-object" in task.runner.command
    assert "evaluation.task_success" in task.required_true
    assert "evaluation.checkpoints.target_released" in task.required_true
