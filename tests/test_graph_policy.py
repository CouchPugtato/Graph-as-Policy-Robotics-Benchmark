from pathlib import Path

import pytest

from gap_isaac.graph_policy import GraphPolicy, GraphPolicyExecutor, NodeResult

ROOT = Path(__file__).resolve().parents[1]


def policy_payload() -> dict:
    return {
        "schema_version": 1,
        "graph_id": "test",
        "start": "attempt",
        "terminals": {"DONE": "succeeded", "FAILED": "failed"},
        "nodes": [
            {
                "id": "attempt",
                "skill": "test-skill",
                "action": "attempt",
                "routes": {"succeeded": "DONE", "failed": "recover"},
            },
            {
                "id": "recover",
                "skill": "test-recovery",
                "action": "recover",
                "routes": {"succeeded": "DONE", "failed": "FAILED"},
            },
        ],
    }


def test_executes_success_route_and_records_skill_trace() -> None:
    policy = GraphPolicy.from_dict(policy_payload())
    result = GraphPolicyExecutor(
        policy,
        {
            "attempt": lambda state: NodeResult(outputs={"value": 7}),
            "recover": lambda state: NodeResult(),
        },
    ).execute()

    assert result.status == "succeeded"
    assert result.state["value"] == 7
    assert [event.node_id for event in result.trace] == ["attempt"]
    assert result.trace[0].skill == "test-skill"


def test_failed_outcome_routes_through_recovery() -> None:
    policy = GraphPolicy.from_dict(policy_payload())
    result = GraphPolicyExecutor(
        policy,
        {
            "attempt": lambda state: NodeResult("failed", "injected"),
            "recover": lambda state: NodeResult("succeeded", "recovered"),
        },
    ).execute()

    assert result.status == "succeeded"
    assert [event.outcome for event in result.trace] == ["failed", "succeeded"]


def test_action_exception_is_routed_and_traced() -> None:
    policy = GraphPolicy.from_dict(policy_payload())

    def fail(_state):
        raise RuntimeError("boom")

    result = GraphPolicyExecutor(
        policy,
        {"attempt": fail, "recover": lambda state: NodeResult("failed")},
    ).execute()

    assert result.status == "failed"
    assert result.trace[0].outcome == "failed"
    assert "RuntimeError: boom" in result.trace[0].detail


def test_rejects_unknown_route_target_and_missing_action() -> None:
    payload = policy_payload()
    payload["nodes"][0]["routes"]["succeeded"] = "missing"
    with pytest.raises(ValueError, match="unknown targets"):
        GraphPolicy.from_dict(payload)

    policy = GraphPolicy.from_dict(policy_payload())
    with pytest.raises(ValueError, match="unavailable"):
        GraphPolicyExecutor(policy, {"attempt": lambda state: NodeResult()})


def test_reference_graph_is_valid_and_pins_upstream_registry() -> None:
    policy = GraphPolicy.from_json(ROOT / "graphs/reference/grocery_single_target.json")

    assert policy.start == "open_gripper"
    assert policy.metadata["geometry_source"] == "oracle-smoke-only"
    assert len(policy.metadata["open_robot_skills_commit"]) == 40
    assert {node.skill for node in policy.nodes.values()} == {
        "grasping-direct-ik",
        "transporting-objects",
    }
