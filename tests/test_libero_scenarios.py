from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gap_harness.models import Task
from gap_isaac.graph_policy import GraphPolicy
from gap_libero.actions import LiberoGraphActions
from gap_libero.scenario_catalog import reference_graph, sequence, task_catalog
from gap_libero.scenario_geometry import estimate
from gap_libero.scenario_runner import ScenarioActions
from gap_libero.scenario_specs import (
    grocery_layout,
    oriented_containment,
    packing_items,
    popcorn_layout,
    validate_case,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scenario", ["fulfillment", "packing", "popcorn"])
def test_reference_policies_and_catalogs_are_bounded_and_local(scenario):
    graph = reference_graph(scenario)
    GraphPolicy.from_dict(graph)
    task = Task.model_validate(task_catalog(scenario))
    assert set(task.required_executed_actions) == {node["action"] for node in graph["nodes"]}
    assert set(task.required_actions) <= set(task.required_executed_actions)
    assert len(graph["nodes"]) <= 64
    assert task.seeds == list(range(10))
    assert task.runner.timeout_s <= 300
    assert graph["metadata"]["perception_backend"] == "local-vlm"
    assert graph["metadata"]["model_generated"] is False


def test_packing_policy_loops_only_after_placement_and_retraction():
    graph = reference_graph("packing")
    nodes = graph["nodes"]
    assert nodes[-2]["action"] == "return_ready"
    assert nodes[-2]["routes"]["succeeded"] == graph["start"]
    assert nodes[0]["routes"]["false"] == nodes[-1]["id"]
    assert nodes[-1]["action"] == "check_all_packed"
    assert "check_placement" in sequence("packing")


@pytest.mark.parametrize("variation", ["xy", "basket-swap", "permutation", "mixed"])
def test_layouts_are_deterministic_and_preserve_designated_seed(variation):
    for seed in range(10):
        plan = grocery_layout(ROOT, seed, variation)
        assert plan == grocery_layout(ROOT, seed, variation)
        assert plan.seed == seed
        assert len(packing_items(seed)) == 2
        assert len(set(packing_items(seed))) == 2


def test_popcorn_model_declares_real_knob_and_dynamic_pan():
    plan = popcorn_layout(0)
    assert next(item for item in plan.objects if item.object_id == "knob").shape == "knob"
    assert next(item for item in plan.objects if item.object_id == "pan").mass_kg == 0.2
    actions = sequence("popcorn")
    assert (
        actions.index("turn_stove_on")
        < actions.index("wait_heating")
        < actions.index("turn_stove_off")
    )
    with pytest.raises(ValueError):
        validate_case("unknown", 0, "mixed")


def test_box_geometry_uses_visible_depth_and_shape_prior_only():
    x, y = np.meshgrid(np.linspace(-0.035, 0.035, 40), np.linspace(-0.025, 0.025, 30))
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = np.column_stack((x.ravel(), y.ravel())) @ rotation.T + [0.5, 0.15]
    cloud = np.column_stack((xy, np.full(len(xy), 0.105)))
    result = estimate(cloud, "chocolate_pudding")
    assert result["position_m"] == pytest.approx([0.5, 0.15, 0.0525], abs=0.001)
    assert result["uses_evaluator_pose"] is False
    with pytest.raises(ValueError):
        estimate(cloud[:5], "chocolate_pudding")


def test_per_item_settle_does_not_require_whole_task_success(tmp_path):
    values = {"target_contained": True, "target_released": True, "target_speed_m_s": 0}
    backend = SimpleNamespace(
        env=SimpleNamespace(
            env=SimpleNamespace(
                containment=lambda: values, scene_plan=SimpleNamespace(target_id="alphabet_soup")
            )
        )
    )
    actions = ScenarioActions(backend, "packing", 0, tmp_path)
    commands = []
    actions.bindings = SimpleNamespace(command=lambda: commands.append(True))
    assert actions.motion("settle_release", {}).outcome == "succeeded"
    assert len(commands) == 10
    values["target_speed_m_s"] = 0.5
    assert actions.motion("settle_release", {}).outcome == "failed"
    assert len(commands) == 130


def test_thin_handle_plan_uses_observed_model_anchor_and_preserves_limits(tmp_path):
    backend = SimpleNamespace(
        workspace=np.array([[-0.35, -0.85, 0.02], [1, 0.85, 1.1]]),
        env=SimpleNamespace(env=SimpleNamespace(scene_plan=SimpleNamespace(target_id="pan"))),
    )
    actions = ScenarioActions(backend, "popcorn", 0, tmp_path)
    actions.vision = SimpleNamespace(
        geometries={
            "pan": {"valid": True, "position_m": [0.7, 0.2, 0.132], "yaw_rad": 0},
            "burner": {"valid": True, "position_m": [0.45, -0.2, 0.115], "yaw_rad": 0},
        }
    )
    plan = actions.plan()
    assert plan.grasp_reference_offset_m == (0.085, 0, 0.014)
    assert plan.actions["descend_to_grasp"].position_m[:2] == pytest.approx([0.665, 0.2])
    assert plan.actions["descend_to_grasp"].position_m[2] < 0.15
    assert plan.maximum_slip_m == 0.015
    assert plan.maximum_grasp_rotation_rad == 0.25


def test_grasp_rotation_guard_rejects_twisting_even_with_stable_anchor():
    actions = object.__new__(LiberoGraphActions)
    actions.plan = SimpleNamespace(
        target_object="pan",
        maximum_slip_m=0.015,
        grasp_reference_offset_m=(0.085, 0, 0.014),
        maximum_grasp_rotation_rad=0.25,
    )
    angle = 0.4
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    actions.backend = SimpleNamespace(
        grasped=lambda name: True,
        target_in_gripper=lambda name, offset: np.zeros(3),
        target_rotation_in_gripper=lambda name: rotation,
    )
    actions.hold_reference = np.zeros(3)
    actions.hold_rotation = np.eye(3)
    actions.missing_contact_frames = 0
    with pytest.raises(RuntimeError, match="rotated"):
        actions.check_hold()


def test_oriented_containment_accepts_fully_contained_horizontal_cylinder():
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert oriented_containment([0, 0, 0.054], rotation, "tomato_sauce", [0, 0, 0], np.eye(3))
    assert not oriented_containment(
        [0, 0.07, 0.054], rotation, "tomato_sauce", [0, 0, 0], np.eye(3)
    )
    assert not oriented_containment([0, 0, 0.03], rotation, "tomato_sauce", [0, 0, 0], np.eye(3))
    assert not oriented_containment([0, 0, 0.16], np.eye(3), "tomato_sauce", [0, 0, 0], np.eye(3))


def test_oriented_containment_checks_box_corners_and_basket_frame():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert oriented_containment([0.5, 0.2, 0.04], rotation, "cream_cheese", [0.5, 0.2, 0], rotation)
    assert not oriented_containment(
        [0.5, 0.26, 0.04], rotation, "cream_cheese", [0.5, 0.2, 0], rotation
    )


@pytest.mark.parametrize("scenario", ["fulfillment", "packing", "popcorn"])
def test_deployed_task_files_match_catalog(scenario):
    import json

    task = ROOT / f"harness/tasks/libero-{scenario}-local-qwen.json"
    assert json.loads(task.read_text()) == task_catalog(scenario)


def test_final_packing_check_rechecks_every_item_not_just_completed_labels(tmp_path):
    values = {
        "alphabet_soup": {
            "target_contained": False,
            "target_released": True,
            "target_speed_m_s": 0,
        },
        "tomato_sauce": {"target_contained": True, "target_released": True, "target_speed_m_s": 0},
    }
    backend = SimpleNamespace(
        env=SimpleNamespace(
            env=SimpleNamespace(
                containment=lambda name: values[name],
                scene_plan=SimpleNamespace(target_id="alphabet_soup"),
            )
        )
    )
    actions = ScenarioActions(backend, "packing", 0, tmp_path)
    actions.completed = list(actions.requested)
    result = actions.check_all_packed({})
    assert result.outcome == "false"
    assert "alphabet_soup" in result.detail
    values["alphabet_soup"]["target_contained"] = True
    assert actions.check_all_packed({}).outcome == "true"
