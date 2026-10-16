import json

import numpy as np
import pytest

from gap_harness.engine import Harness
from gap_harness.feedback import build_trial_feedback, compact_feedback, trace_matches_graph
from gap_harness.models import CheckpointSet, Task
from gap_isaac.graph_policy import (
    GraphExecutionBoundExceeded,
    GraphPolicy,
    GraphPolicyExecutor,
    NodeResult,
)
from gap_libero.scenario_catalog import reference_workflow
from gap_libero.scenario_geometry import estimate
from gap_libero.scenario_validation import prepare_reference


@pytest.mark.parametrize("scenario", ["fulfillment", "packing", "popcorn"])
def test_reference_compiles_through_generation_validator(scenario):
    bundle = reference_workflow(scenario)
    assert bundle["graph"]["metadata"]["reference_policy"] is True
    assert bundle["graph"]["metadata"]["model_generated"] is False
    assert len(bundle["graph"]["nodes"]) <= 64
    assert bundle["graph"]["metadata"]["subgraph_membership"]


@pytest.mark.parametrize("scenario", ["fulfillment", "packing", "popcorn"])
def test_complete_generation_path_with_offline_agent_fixture(scenario, tmp_path):
    from pathlib import Path

    from gap_harness.models import Subgraph, Topology

    bundle = reference_workflow(scenario)

    class Provider:
        model = "offline-reference-fixture"

        def complete(self, instructions, context, output_model):
            payload = (
                bundle["topology"]
                if output_model is Topology
                else (
                    bundle["subgraphs"][0]
                    if output_model is Subgraph
                    else {"checkpoints": bundle["checkpoints"]}
                )
            )
            return {"status": "completed", "text": json.dumps(payload), "usage": {}}

    task = Path(__file__).resolve().parents[1] / f"harness/tasks/libero-{scenario}-local-qwen.json"
    harness = Harness(task, tmp_path / "session", Provider())
    version = harness.generate()
    assert json.loads((version / "validation.json").read_text())["ok"]
    assert (version / "workflow.json").is_file()


def loop_example(max_steps=160, exhaust=False):
    graph = {
        "schema_version": 1,
        "graph_id": "loop-test",
        "start": "visit",
        "nodes": [
            {
                "id": "visit",
                "skill": "test",
                "action": "more",
                "routes": {"true": "visit", "false": "done", "failed": "abort"},
            }
        ],
        "terminals": {"done": "succeeded", "abort": "failed"},
        "metadata": {"subgraph_membership": {"visit": "phase"}, "node_kinds": {"visit": "router"}},
    }

    def action(state):
        state["count"] = state.get("count", 0) + 1
        return NodeResult("true" if exhaust or state["count"] < 80 else "false")

    executor = GraphPolicyExecutor(
        GraphPolicy.from_dict(graph), {"more": action}, max_steps=max_steps
    )
    bound = False
    try:
        result = executor.execute()
    except GraphExecutionBoundExceeded as error:
        result = error.result
        bound = True
    receipt = {
        "ok": not bound,
        "seed": 0,
        "graph_sha256": "test",
        "graph": result.as_dict(),
        "evaluation": {"verified": not bound},
    }
    if bound:
        receipt["graph"]["execution_bound_exceeded"] = True
    task = Task.model_validate(
        {
            "task_id": "loop-test",
            "instruction": "test",
            "seeds": [0],
            "skills": [
                {
                    "name": "test",
                    "description": "test",
                    "instructions": "test",
                    "actions": [
                        {
                            "name": "more",
                            "description": "test",
                            "kind": "router",
                            "outcomes": ["true", "false", "failed"],
                        },
                        {"name": "unused", "description": "unused"},
                    ],
                }
            ],
            "checkpoint_evidence": ["evaluation.verified"],
            "required_true": ["evaluation.verified"],
            "required_executed_actions": ["more"],
            "max_trace_steps": max_steps,
        }
    )
    checks = CheckpointSet.model_validate(
        {
            "checkpoints": [
                {
                    "subgraph": "phase",
                    "name": "verified",
                    "rationale": "test",
                    "evidence": "evaluation.verified",
                    "expected": "true",
                }
            ]
        }
    )
    return graph, receipt, task, checks


def test_long_trace_and_exhaustion_branch_are_valid_feedback():
    graph, receipt, task, checks = loop_example()
    assert not trace_matches_graph(receipt, graph)
    feedback = build_trial_feedback(receipt, task, graph, checks, "test", 0)
    assert feedback["success"] and feedback["evaluated"]
    assert feedback["failures"] == []


def test_runtime_action_requirement_cannot_be_bypassed_by_claiming_success():
    graph, receipt, task, checks = loop_example()
    task.required_executed_actions.append("unused")
    feedback = build_trial_feedback(receipt, task, graph, checks, "test", 0)
    assert feedback["evaluated"] and not feedback["success"]
    assert feedback["missing_executed_actions"] == ["unused"]
    assert any(failure.get("kind") == "execution_guard" for failure in feedback["failures"])


def test_graph_visit_limit_is_policy_feedback_not_infrastructure():
    graph, receipt, task, checks = loop_example(max_steps=100, exhaust=True)
    feedback = build_trial_feedback(receipt, task, graph, checks, "test", 0, process_ok=False)
    assert feedback["evaluated"] and not feedback["success"]
    assert feedback["failure_kind"] == "policy"
    assert any("visit budget" in failure["detail"] for failure in feedback["failures"])
    receipt["graph"]["trace"].pop()
    assert not trace_matches_graph(receipt, graph, 100)


def test_visible_cylinder_cap_can_constrain_geometry_without_sidewall():
    x, y = np.meshgrid(np.linspace(-0.034, 0.034, 35), np.linspace(-0.034, 0.034, 35))
    mask = x * x + y * y <= 0.034**2
    cloud = np.column_stack((x[mask] + 0.5, y[mask] + 0.1, np.full(mask.sum(), 0.105)))
    fit = estimate(cloud, "tomato_sauce", silhouette_points=cloud)
    assert fit["position_m"] == pytest.approx([0.5, 0.1, 0.0525], abs=0.001)
    assert fit["cap_fit"]["rms_error_m"] < 0.002
    with pytest.raises(ValueError):
        estimate(cloud[cloud[:, 0] > 0.52], "tomato_sauce")


def test_reference_session_is_explicit_and_immutable(tmp_path):
    harness, version = prepare_reference("packing", tmp_path, [1])
    assert harness.provider is None
    assert harness.task.seeds == [1]
    assert json.loads((version / "lineage.json").read_text())["model_generated"] is False
    assert (version / "validation.json").is_file()
    with pytest.raises(FileExistsError):
        prepare_reference("packing", tmp_path, [1])


def test_new_optional_limits_do_not_change_legacy_session_snapshot(tmp_path):
    _, _, task, _ = loop_example()
    document = task.model_dump()
    document.pop("max_trace_steps")
    document.pop("required_executed_actions")
    path = tmp_path / "task.json"
    path.write_text(json.dumps(document))
    harness = Harness(path, tmp_path / "session")
    assert harness.snapshot["task"] == document


def test_compacted_feedback_preserves_per_item_evidence():
    graph, receipt, task, checks = loop_example()
    receipt["evaluation"]["scenario_evidence"] = {"items": {"one": {"target_contained": False}}}
    feedback = build_trial_feedback(receipt, task, graph, checks, "test", 0)
    compacted = compact_feedback({"records": [feedback]}, max_characters=100)
    assert compacted["records"][0]["scenario_evidence"]["items"]["one"]["target_contained"] is False


def test_packing_bin_prior_is_shared_by_scene_and_image_fit():
    from pathlib import Path

    from gap_libero.scenario_specs import PACKING_BASKET_SIZE, grocery_layout
    from gap_libero.vision import estimate_geometry

    for seed in range(10):
        scene = grocery_layout(
            Path(__file__).resolve().parents[1], seed, "mixed", PACKING_BASKET_SIZE
        )
        assert (
            next(item.size_m for item in scene.objects if item.object_id == "basket")
            == PACKING_BASKET_SIZE
        )
    edge = np.linspace(-1, 1, 100)
    x, y = PACKING_BASKET_SIZE[0] / 2, PACKING_BASKET_SIZE[1] / 2
    ring = np.concatenate(
        [np.column_stack((edge * x, np.full(100, sign * y))) for sign in (-1, 1)]
        + [np.column_stack((np.full(100, sign * x), edge * y)) for sign in (-1, 1)]
    )
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = ring @ rotation.T + [0.5, 0.1]
    fit = estimate_geometry(
        np.column_stack((xy, np.full(len(xy), 0.17))), "basket", basket_size=PACKING_BASKET_SIZE
    )
    assert fit["position_m"][:2] == pytest.approx([0.5, 0.1], abs=0.001)


def test_cli_selects_new_scenario_without_model_calls(monkeypatch, capsys):
    import sys

    from gap_harness import cli

    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(sys, "argv", ["gap-harness", "check", "--scenario", "popcorn"])
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["task"] == "libero-popcorn-local-qwen"
    assert report["seeds"] == list(range(10))
    assert report["network_requests"] == 0


def test_release_uses_encoder_commands_and_clears_one_sided_contact(tmp_path):
    from types import SimpleNamespace

    from gap_libero.scenario_runner import ScenarioActions

    contact = SimpleNamespace(geom1=0, geom2=2)
    data = SimpleNamespace(
        contact=[contact],
        ncon=1,
        get_joint_qpos=lambda name: 0.036 if name == "left" else -0.036,
        get_site_xmat=lambda site: np.eye(3),
    )
    model = SimpleNamespace(
        geom_name2id={"target": 0, "left": 1, "right": 2}.__getitem__,
        joint_name2id={"left": 0, "right": 1}.__getitem__,
        jnt_range=np.array([[0, 0.04], [-0.04, 0]]),
    )
    gripper = SimpleNamespace(
        joints=["left", "right"],
        contact_geoms=["left", "right"],
        important_geoms={"left_finger": ["left"], "right_finger": ["right"]},
        important_sites={"grip_site": "eef"},
        current_action=np.zeros(2),
    )
    env = SimpleNamespace(
        robots=[SimpleNamespace(gripper=gripper)],
        sim=SimpleNamespace(data=data, model=model),
        env=SimpleNamespace(
            scene_plan=SimpleNamespace(target_id="alphabet_soup"),
            objects_dict={"alphabet_soup": SimpleNamespace(contact_geoms=["target"])},
        ),
    )
    actions = ScenarioActions(SimpleNamespace(env=env), "packing", 0, tmp_path)
    retract = SimpleNamespace(position_m=(0.5, 0.1, 0.28))
    bindings = SimpleNamespace(
        hold_position=np.zeros(3), plan=SimpleNamespace(actions={"retract_from_basket": retract})
    )

    def command():
        if bindings.hold_position[0] > 0.002:
            data.contact = []
            data.ncon = 0

    bindings.command = command
    actions.bindings = bindings
    result = actions.motion("release_object", {})
    assert result.outcome == "succeeded"
    assert bindings.hold_position == pytest.approx([0.003, 0, 0])
    assert retract.position_m == pytest.approx([0.503, 0.1, 0.28])
    assert np.max(np.abs(gripper.current_action)) <= 1
