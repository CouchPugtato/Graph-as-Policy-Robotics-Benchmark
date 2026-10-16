import copy

import numpy as np
import pytest

from gap_harness.models import CheckpointSet, Subgraph, Task, Topology
from gap_harness.workflow import compile_workflow
from gap_isaac.graph_policy import GraphPolicy, GraphPolicyExecutor, NodeResult
from gap_libero.vision import color_mask, estimate_geometry, fit_basket_walls
from gap_libero.vision_workflow import adapt_workflow, vision_task


def test_color_masks_use_pixels_and_reject_missing_targets():
    image = np.array([[[20, 80, 200], [35, 160, 60], [190, 20, 20], [80, 80, 80]]],
                     dtype=np.uint8)
    assert color_mask(image, "alphabet_soup").tolist() == [[True, False, False, False]]
    assert color_mask(image, "basket").tolist() == [[False, True, False, False]]
    assert not color_mask(np.zeros_like(image), "alphabet_soup").any()


def test_cylinder_fit_uses_only_visible_world_points():
    angle, height = np.meshgrid(np.linspace(-2.7, -.4, 80), np.linspace(.02, .109, 20))
    points = np.column_stack((.5 + .035 * np.cos(angle.ravel()),
                              .15 + .035 * np.sin(angle.ravel()), height.ravel()))
    result = estimate_geometry(points, "alphabet_soup")
    assert result["position_m"][:2] == pytest.approx([.5, .15], abs=.001)
    assert not result["uses_evaluator_pose"]


def test_occluded_cylinder_requires_independent_top_rim_evidence():
    from gap_isaac.cylinder_center import fit_center

    angle, height = np.meshgrid(np.linspace(-.35, .35, 30), np.linspace(.02, .11, 15))
    points = np.column_stack((.5 + .035 * np.cos(angle.ravel()),
                              .15 + .035 * np.sin(angle.ravel()), height.ravel()))
    with pytest.raises(ValueError, match="insufficiently constrained"):
        fit_center(points, [.5, .15])
    angle = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    rim = np.column_stack((.5 + .035 * np.cos(angle), .15 + .035 * np.sin(angle),
                           np.full(len(angle), .11)))
    fit = fit_center(points, [.5, .15], top_rim=rim)
    assert fit["sidewall_angular_span_rad"] < np.pi / 3
    assert fit["top_rim_support"]["points"] == 32
    assert fit["center_xy_m"] == pytest.approx([.5, .15], abs=.001)
    rim[:, :2] += .03
    with pytest.raises(ValueError):
        fit_center(points, [.5, .15], top_rim=rim)


def test_basket_fit_uses_opposite_rims_when_one_wall_is_hidden():
    x, z = np.meshgrid(np.linspace(-.09, .09, 40), np.linspace(.03, .12, 20))
    wall = np.column_stack((x.ravel(), np.full(x.size, -.085), z.ravel()))
    x, y = np.meshgrid(np.linspace(-.115, .115, 100), np.linspace(-.095, .095, 90))
    ring = (np.abs(x) >= .106) | (np.abs(y) >= .086)
    rim = np.column_stack((x[ring], y[ring], np.full(ring.sum(), .17)))
    cloud = np.concatenate((wall, rim)) + [.4, -.3, 0]
    center, angle, report = fit_basket_walls(cloud, [.402, -.302], .003)
    assert center == pytest.approx([.4, -.3], abs=.004)
    assert angle == pytest.approx(0, abs=.02)
    assert min(report["opposite_rim_support"][0]) >= 20
    with pytest.raises(ValueError, match="independent plane support"):
        fit_basket_walls(wall + [.4, -.3, 0], [.4, -.3], 0)


def test_basket_rim_reconstruction():
    edge = np.linspace(-1, 1, 100)
    local = np.concatenate([np.column_stack((edge * .12, np.full(100, side * .10)))
                            for side in (-1, 1)] + [
        np.column_stack((np.full(100, side * .12), edge * .10)) for side in (-1, 1)])
    angle = .4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = local @ rotation.T + [.4, -.3]
    result = estimate_geometry(np.column_stack((xy, np.full(len(xy), .16))), "basket")
    assert result["position_m"][:2] == pytest.approx([.4, -.3], abs=.001)
    assert np.cos(2 * result["yaw_rad"]) == pytest.approx(np.cos(2 * angle), abs=.02)


@pytest.mark.parametrize("points", [np.zeros((0, 3)), np.full((50, 3), np.nan),
                                    np.full((50, 3), -1)])
def test_bad_geometry_fails_closed(points):
    with pytest.raises(ValueError):
        estimate_geometry(points, "alphabet_soup")


@pytest.mark.parametrize("found", [True, False])
def test_real_boolean_routing_outputs_are_traced(found):
    graph = {"schema_version": 1, "graph_id": "vision-test", "start": "found",
        "metadata": {"node_kinds": {"found": "router"}},
        "nodes": [{"id": "found", "skill": "vision", "action": "check_target",
                   "routes": {"true": "DONE", "false": "ABORT", "failed": "ABORT"}}],
        "terminals": {"DONE": "succeeded", "ABORT": "failed"}}
    result = GraphPolicyExecutor(GraphPolicy.from_dict(graph), {
        "check_target": lambda state: NodeResult("true" if found else "false",
                                                 outputs={"target_found": found})}).execute()
    assert result.status == ("succeeded" if found else "failed")
    assert result.trace[0].outputs == {"target_found": found}
    assert result.state["target_found"] is found


def test_adaptation_preserves_source_and_enforces_vision_prerequisites():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    task = json.loads((root / "harness/tasks/libero-grocery.json").read_text())
    action_groups = [[a["name"] for a in skill["actions"]] for skill in task["skills"]]
    specs, subgraphs, checks = [], [], []
    for index, names in enumerate(action_groups):
        name = f"phase{index}"
        specs.append({"name": name, "skill": task["skills"][index]["name"],
            "description": "phase", "inputs": [], "outputs": [], "exits": ["done"],
            "on_error": "error", "routes": [{"outcome": "done", "target":
                "phase1" if index == 0 else "DONE"}, {"outcome": "error", "target": "ABORT"}]})
        subgraphs.append({"name": name, "rationale": "phase", "start": names[0],
            "nodes": [{"name": action, "kind": "tool", "action": action,
                       "routes": [{"outcome": "succeeded", "target":
                           names[i+1] if i+1 < len(names) else "done"},
                           {"outcome": "failed", "target": "error"}]}
                      for i, action in enumerate(names)]})
        checks.append({"subgraph": name, "name": "verified", "rationale": "check",
                       "evidence": task["checkpoint_evidence"][index], "expected": "true"})
    source = {"topology": {"graph_id": "test", "rationale": "test", "start": "phase0",
        "subgraphs": specs, "terminals": [{"name": "DONE", "status": "succeeded"},
                                          {"name": "ABORT", "status": "failed"}]},
        "subgraphs": subgraphs, "checkpoints": checks}
    original = copy.deepcopy(source)
    adapted = adapt_workflow(source)
    assert source == original
    graph = compile_workflow(Topology.model_validate(adapted["topology"]),
        [Subgraph.model_validate(value) for value in adapted["subgraphs"]],
        Task.model_validate(vision_task(task)),
        CheckpointSet.model_validate({"checkpoints": adapted["checkpoints"]}))
    assert list(graph["metadata"]["node_kinds"].values()).count("router") == 4
    assert len(graph["nodes"]) == 21
    assert set(graph["metadata"]["subgraph_membership"].values()) == {"phase0", "phase1"}
    assert adapted["subgraphs"][0]["start"] == "observe_scene"
    assert adapt_workflow(adapted) == adapted
    local = adapt_workflow(source, "local-vlm")
    local_graph = compile_workflow(Topology.model_validate(local["topology"]),
        [Subgraph.model_validate(value) for value in local["subgraphs"]],
        Task.model_validate(vision_task(task, "local-vlm")),
        CheckpointSet.model_validate({"checkpoints": local["checkpoints"]}))
    assert len(local_graph["nodes"]) == 23
    assert set(local_graph["metadata"]["subgraph_membership"].values()) == {"phase0", "phase1"}
    from gap_harness.workflow import validate_topology
    rejected = copy.deepcopy(local["topology"])
    rejected["subgraphs"][0]["skill"] = "perceiving-proxy-objects"
    with pytest.raises(ValueError, match="shared perception tools"):
        validate_topology(Topology.model_validate(rejected),
                          Task.model_validate(vision_task(task, "local-vlm")))


def test_image_derived_motion_plan_needs_no_simulator_object_pose_access():
    from types import SimpleNamespace

    from gap_libero.grocery_comparison import build_action_plan

    backend = SimpleNamespace(workspace=np.array([[-.35, -.85, .02], [1, .85, 1.1]]),
                              max_steps=1000)
    perception = {"alphabet_soup": {"position_m": [.5, .2, .055]},
                  "basket": {"position_m": [.4, -.25, 0], "yaw_rad": .4}}
    plan = build_action_plan(backend, 0, perception)
    assert plan.actions["descend_to_grasp"].position_m == pytest.approx([.5, .2, .09])
    assert plan.actions["move_over_basket"].position_m[:2] == pytest.approx([.4, -.25])
