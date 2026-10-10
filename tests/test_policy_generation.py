import copy
import json
from pathlib import Path

import pytest

from gap_isaac.policy_generation import (
    MODEL,
    accept,
    graph_schema,
    load_json,
    metadata,
    prepare,
    sha256,
    validate_generated_graph,
    verify_accepted,
    write_new,
)

ROOT = Path(__file__).resolve().parents[1]


def payload():
    # Synthetic validator fixture, not evidence of model generation.
    result = json.loads((ROOT / "graphs/reference/grocery_single_target.json").read_text())
    result["metadata"] = metadata()
    result["graph_id"] = "synthetic-validation-fixture"
    return result


def package(tmp_path):
    for name in ("prompt.md", "schema.json", "skill-definitions.md"):
        (tmp_path / name).write_text("synthetic fixture")
    write_new(tmp_path / "package.json", {
        "schema_version": 1, "requested_model": MODEL,
        "file_sha256": {name: sha256((tmp_path / name).read_bytes())
                        for name in ("prompt.md", "schema.json", "skill-definitions.md")},
    })
    write_new(tmp_path / "candidate.json", payload())
    return tmp_path


def test_validates_and_fake_executes_without_models():
    from gap_isaac.graph_policy import GraphPolicyExecutor, NodeResult
    policy = validate_generated_graph(payload())
    actions = {node.action: lambda state: NodeResult() for node in policy.nodes.values()}
    result = GraphPolicyExecutor(policy, actions, max_steps=10).execute()
    assert result.status == "succeeded"
    assert len(result.trace) == 10


@pytest.mark.parametrize("mutation", [
    "unknown", "duplicate", "skill", "cycle", "early_success", "failure_recovery",
    "pose", "code", "metadata", "terminal", "id_overlap", "bool_version", "bad_order",
    "unreachable", "bad_id", "route_type", "node_type", "missing_action",
])
def test_rejects_unsafe_or_invalid_graphs(mutation):
    data = copy.deepcopy(payload())
    node = data["nodes"][0]
    if mutation == "unknown":
        node["action"] = "execute_python"
    elif mutation == "duplicate":
        data["nodes"][1]["action"] = node["action"]
    elif mutation == "skill":
        node["skill"] = "invented"
    elif mutation == "cycle":
        data["nodes"][-1]["routes"]["succeeded"] = data["start"]
    elif mutation == "early_success":
        node["routes"]["succeeded"] = "SUCCEEDED"
    elif mutation == "failure_recovery":
        node["routes"]["failed"] = "align_over_target"
    elif mutation == "pose":
        node["position"] = [0, 0, 0]
    elif mutation == "code":
        data["python"] = "print('no')"
    elif mutation == "metadata":
        data["metadata"]["geometry_source"] = "oracle-smoke-only"
    elif mutation == "terminal":
        data["terminals"]["FAILED"] = "succeeded"
    elif mutation == "id_overlap":
        node["id"] = "FAILED"
    elif mutation == "bool_version":
        data["schema_version"] = True
    elif mutation == "bad_order":
        data["start"] = "release_object"
    elif mutation == "unreachable":
        data["nodes"][-2]["routes"]["succeeded"] = "SUCCEEDED"
    elif mutation == "bad_id":
        node["id"] = "../evil"
    elif mutation == "route_type":
        node["routes"]["succeeded"] = []
    elif mutation == "node_type":
        data["nodes"][0] = None
    elif mutation == "missing_action":
        data["nodes"].pop()
    with pytest.raises(ValueError):
        validate_generated_graph(data)


def test_ids_and_serialization_order_are_model_selected():
    data = payload()
    mapping = {node["id"]: f"step_{i}" for i, node in enumerate(data["nodes"])}
    data["start"] = mapping[data["start"]]
    for node in data["nodes"]:
        node["id"] = mapping[node["id"]]
        node["routes"]["succeeded"] = mapping.get(
            node["routes"]["succeeded"], node["routes"]["succeeded"])
    data["nodes"].reverse()
    assert len(validate_generated_graph(data).nodes) == 10


def test_freezes_acceptance_and_detects_tampering(tmp_path):
    bundle = package(tmp_path)
    path = accept(bundle, MODEL, "synthetic unit test; not model evidence")
    receipt = verify_accepted(path)
    assert receipt["model_verified"] is False
    assert receipt["network_requests"] == 0
    with pytest.raises(ValueError, match="already exist"):
        accept(bundle, MODEL, "repeat")
    altered = payload()
    altered["graph_id"] = "changed"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="provenance mismatch"):
        verify_accepted(path)


def test_rejects_changed_prompt_and_non_astra(tmp_path):
    bundle = package(tmp_path)
    with pytest.raises(ValueError, match="only declared"):
        accept(bundle, "other-model", "test")
    (bundle / "prompt.md").write_text("changed")
    with pytest.raises(ValueError, match="package changed"):
        accept(bundle, MODEL, "test")


def test_rejects_duplicate_json_keys_and_oversized_files(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"nodes": [], "nodes": []}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_json(path)
    path.write_text(" " * (256 * 1024 + 1))
    with pytest.raises(ValueError, match="limit"):
        load_json(path)


def test_schema_has_no_code_or_parameters():
    schema = graph_schema()
    assert schema["additionalProperties"] is False
    node = schema["properties"]["nodes"]["items"]
    assert set(node["properties"]) == {"id", "skill", "action", "routes"}


def test_prepare_fails_without_pinned_checkout(tmp_path):
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):
        prepare(tmp_path)


def test_simulator_runner_rejects_unaccepted_graph_before_isaac(tmp_path):
    import subprocess
    import sys

    bundle = package(tmp_path)
    # Ensure ordinary path checks pass; this is never opened as a USD stage.
    stage = tmp_path / "dummy.usd"
    stage.write_text("synthetic preflight fixture")
    result = subprocess.run([
        sys.executable, str(ROOT / "isaac/validate_graph_pick_place.py"),
        "--stage", str(stage), "--task", str(ROOT / "configs/tasks/grocery_fulfillment.yaml"),
        "--graph", str(bundle / "candidate.json"),
        "--target-geometry", "unused.json", "--basket-geometry", "unused.json",
        "--grasp-result", "unused.json",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "accepted.json artifact" in result.stderr
    assert "No module named 'isaacsim'" not in result.stderr
