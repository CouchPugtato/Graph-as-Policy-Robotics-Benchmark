from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from gap_harness.engine import Harness, digest, read
from gap_harness.feedback import aggregate_feedback, build_trial_feedback, compact_feedback
from gap_harness.models import CheckpointSet, Subgraph, Task, Topology
from gap_harness.progress import Progress
from gap_harness.provider import OpenAIResponses
from gap_harness.workflow import compile_workflow, validate_topology
from gap_isaac.graph_policy import GraphPolicy, GraphPolicyExecutor, NodeResult

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_api_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def task_payload():
    return {"task_id": "unit-test", "instruction": "Perform the configured task",
            "skills": [{"name": "test-skill", "description": "test",
                        "instructions": "Only use the action catalog",
                        "actions": [{"name": "act", "description": "test action"}]}],
            "seeds": [0, 1], "checkpoint_evidence": ["evaluation.verified"],
            "required_true": ["evaluation.verified"], "required_actions": ["act"]}


def topology_payload(count=1):
    return {"graph_id": "test", "rationale": "test", "start": "phase0",
            "terminals": [{"name": "DONE", "status": "succeeded"},
                          {"name": "ABORT", "status": "failed"}],
            "subgraphs": [{"name": f"phase{i}", "skill": "test-skill",
                           "description": "one phase", "inputs": [], "outputs": [],
                           "exits": ["ready"], "on_error": "error",
                           "routes": [{"outcome": "ready", "target":
                                       f"phase{i+1}" if i < count - 1 else "DONE"},
                                      {"outcome": "error", "target": "ABORT"}]}
                          for i in range(count)]}


def subgraph_payload(name):
    return {"name": name, "rationale": "one action", "start": "step",
            "nodes": [{"name": "step", "kind": "tool", "action": "act",
                       "routes": [{"outcome": "succeeded", "target": "ready"},
                                  {"outcome": "failed", "target": "error"}]}]}


def checks_payload(count=1):
    return {"checkpoints": [{"subgraph": f"phase{i}", "name": "verified",
                             "rationale": "check task evidence",
                             "evidence": "evaluation.verified", "expected": "true"}
                            for i in range(count)]}


def compiled(count=1):
    return compile_workflow(Topology.model_validate(topology_payload(count)),
        [Subgraph.model_validate(subgraph_payload(f"phase{i}")) for i in range(count)],
        Task.model_validate(task_payload()), CheckpointSet.model_validate(checks_payload(count)))


class FakeProvider:
    model = "offline-test"

    def __init__(self, count=2):
        self.count, self.calls = count, []

    def complete(self, instructions, context, output_model):
        self.calls.append((output_model.__name__, context))
        if output_model is Topology:
            payload = topology_payload(self.count)
        elif output_model is Subgraph:
            payload = subgraph_payload(context["assignment"]["name"])
        else:
            payload = checks_payload(self.count)
        return {"status": "completed", "text": json.dumps(payload),
                "usage": {"input_tokens": 1, "output_tokens": 1}}


@pytest.fixture
def harness(tmp_path):
    task = tmp_path / "task.json"
    task.write_text(json.dumps(task_payload()))
    return Harness(task, tmp_path / "session", FakeProvider())


@pytest.mark.parametrize("count", [1, 2, 5])
def test_orchestrator_controls_agent_count(harness, count):
    harness.provider.count = count
    version = harness.generate()
    roles = [role for role, _ in harness.provider.calls]
    assert roles == ["Topology"] + ["Subgraph"] * count + ["CheckpointSet"]
    assert read(version / "validation.json")["skill_agent_count"] == count


def test_generation_reports_agents_validation_and_usage_without_prompt_content(harness):
    output = io.StringIO()
    harness.progress = Progress(stream=output)
    harness.task.instruction = "private task instruction not for the terminal"
    harness.generate()
    text = output.getvalue()
    assert "coordinator: started" in text
    assert "skill agent 1/2" in text and "skill agent 2/2" in text
    assert "checkpoint-agent: started" in text
    assert "running final structural validation" in text
    assert "generation complete; simulation not yet evaluated" in text
    assert "calls 4/48" in text and "reported session tokens 8" in text
    assert "private task instruction not for the terminal" not in text
    assert len(harness.provider.calls) == 4


def test_cli_progress_uses_stderr_and_preserves_json_output(harness, monkeypatch, capsys):
    import sys

    from gap_harness import cli

    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli, "OpenAIResponses", lambda model, **kwargs: harness.provider)
    monkeypatch.setattr(sys, "argv", ["gap-harness", "generate", "--live", "--task",
        str(harness.task_path), "--session", str(harness.session)])
    cli.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["accepted"] is False
    assert "revision attempt 1/1" in captured.err
    assert "coordinator: started" in captured.err
    assert "limits:" in captured.err
    assert len(harness.provider.calls) == 4


def test_compiler_preserves_execution_routes_and_membership():
    graph = compiled(3)
    result = GraphPolicyExecutor(GraphPolicy.from_dict(graph),
                                {"act": lambda state: NodeResult()}).execute()
    assert result.status == "succeeded"
    assert [event.node_id for event in result.trace] == [f"phase{i}__step" for i in range(3)]
    assert len(graph["metadata"]["subgraph_membership"]) == 3
    failed = GraphPolicyExecutor(GraphPolicy.from_dict(graph),
                                {"act": lambda state: NodeResult("failed")}).execute()
    assert failed.status == "failed"
    assert len(failed.trace) == 1


@pytest.mark.parametrize("mutation", ["unknown_skill", "unknown_route", "duplicate", "agent_cap"])
def test_invalid_topology_rejected(mutation):
    topology = topology_payload()
    if mutation == "unknown_skill":
        topology["subgraphs"][0]["skill"] = "invented"
    elif mutation == "unknown_route":
        topology["subgraphs"][0]["routes"][0]["target"] = "missing"
    elif mutation == "duplicate":
        topology["subgraphs"].append(topology["subgraphs"][0])
    with pytest.raises(ValueError):
        validate_topology(Topology.model_validate(topology), Task.model_validate(task_payload()),
                          max_subgraphs=0 if mutation == "agent_cap" else 8)


def test_input_contract_rejects_unavailable_port():
    topology = topology_payload()
    topology["subgraphs"][0]["inputs"] = [{"name": "cloud", "type": "PointCloud"}]
    with pytest.raises(ValueError, match="unavailable"):
        validate_topology(Topology.model_validate(topology), Task.model_validate(task_payload()))


def test_input_contract_accepts_predecessor_output():
    topology = topology_payload(2)
    port = {"name": "cloud", "type": "PointCloud"}
    topology["subgraphs"][0]["outputs"] = [port]
    topology["subgraphs"][1]["inputs"] = [port]
    validate_topology(Topology.model_validate(topology), Task.model_validate(task_payload()))


@pytest.mark.parametrize("mutation", ["action", "kind", "route", "outcome", "checkpoint"])
def test_compiler_rejects_invented_capabilities(mutation):
    sg = subgraph_payload("phase0")
    checks = checks_payload()
    if mutation == "action":
        sg["nodes"][0]["action"] = "run_shell"
    elif mutation == "kind":
        sg["nodes"][0]["kind"] = "script"
    elif mutation == "route":
        sg["nodes"][0]["routes"][1]["target"] = "ready"
    elif mutation == "outcome":
        sg["nodes"][0]["routes"].pop()
    else:
        checks["checkpoints"][0]["evidence"] = "invented.success"
    with pytest.raises(ValueError):
        compile_workflow(Topology.model_validate(topology_payload()),
            [Subgraph.model_validate(sg)], Task.model_validate(task_payload()),
            CheckpointSet.model_validate(checks))


def test_versions_are_retained_and_feedback_reaches_agents(harness):
    first = harness.generate()
    original = (first / "graph.json").read_bytes()
    feedback = {"trials": 2, "successes": 0, "success_rate": 0, "detail": "target slipped"}
    (first / "feedback.json").write_text(json.dumps(feedback))
    second = harness.generate()
    assert first.name == "v0001" and second.name == "v0002"
    assert (first / "graph.json").read_bytes() == original
    assert read(second / "lineage.json")["parent"] == first.name
    assert harness.provider.calls[4][1]["previous"]["feedback"] == feedback
    assert harness.provider.calls[5][1]["previous"]["workflow"]


def test_failed_attempt_and_raw_output_are_preserved(harness):
    harness.provider.count = 9
    with pytest.raises(ValueError, match="budget"):
        harness.generate()
    first = harness.session / "v0001"
    assert read(first / "validation.json")["ok"] is False
    assert list(first.glob("calls/*/response.json"))
    harness.provider.count = 1
    assert harness.generate().name == "v0002"


def test_changed_task_requires_new_session(harness):
    harness.generate()
    payload = task_payload()
    payload["required_true"] = ["evaluation.verified"]
    payload["instruction"] = "a different task"
    harness.task_path.write_text(json.dumps(payload))
    other = Harness(harness.task_path, harness.session, FakeProvider())
    with pytest.raises(ValueError, match="changed"):
        other.generate()


def test_session_call_budget_and_lock(harness):
    harness.max_calls = 1
    with pytest.raises(ValueError, match="budget"):
        harness.generate()
    assert len(list(harness.session.glob("v*/calls/*/request.json"))) == 1
    assert not (harness.session / ".lock").exists()
    (harness.session / ".lock").write_text("another process")
    with pytest.raises(FileExistsError):
        harness.generate()


def receipt_for(graph, graph_hash, seed=0):
    trace = GraphPolicyExecutor(GraphPolicy.from_dict(graph),
                               {"act": lambda state: NodeResult()}).execute().as_dict()
    return {"ok": True, "seed": seed, "graph_sha256": graph_hash, "graph": trace,
            "evaluation": {"verified": True}, "initial_state": {"grasped": False},
            "trajectory": [{"action": "act", "grasped": True, "contacts": ["finger"]}]}


def test_feedback_requires_fixed_checks_graph_and_seed_binding():
    graph = compiled()
    task = Task.model_validate(task_payload())
    checks = CheckpointSet.model_validate(checks_payload())
    receipt = receipt_for(graph, "hash")
    def evaluate():
        return build_trial_feedback(receipt, task, graph, checks, "hash", 0)
    assert evaluate()["success"]
    assert evaluate()["phase_state_changes"][0]["before"] == {"grasped": False}
    receipt["evaluation"]["verified"] = 1
    assert not evaluate()["success"]
    receipt["evaluation"]["verified"] = True
    receipt["graph_sha256"] = "wrong"
    assert not evaluate()["success"]
    receipt["graph_sha256"] = "hash"
    receipt["seed"] = 1
    assert not evaluate()["success"]


def test_feedback_attributes_failures_to_subgraphs():
    graph = compiled()
    receipt = receipt_for(graph, "hash")
    terminal = graph["nodes"][0]["routes"]["failed"]
    receipt["graph"]["trace"][0].update(outcome="failed", detail="slipped", next_node=terminal)
    receipt["graph"]["terminal"] = terminal
    receipt["graph"]["status"] = "failed"
    trial = build_trial_feedback(receipt, Task.model_validate(task_payload()), graph,
        CheckpointSet.model_validate(checks_payload()), "hash", 0)
    assert aggregate_feedback([trial])["failures_by_subgraph"] == {"phase0": 1}


def test_failed_postcondition_is_reported_to_its_skill_agent():
    graph = compiled()
    receipt = receipt_for(graph, "hash")
    receipt["evaluation"]["verified"] = False
    trial = build_trial_feedback(receipt, Task.model_validate(task_payload()), graph,
        CheckpointSet.model_validate(checks_payload()), "hash", 0)
    assert trial["failures"][0]["kind"] == "checkpoint"
    assert trial["failures"][0]["subgraph"] == "phase0"
    assert not trial["success"]


def test_large_feedback_keeps_failed_phase_and_preserves_raw_evidence():
    feedback = {"trials": 1, "successes": 0, "records": [{"seed": 0,
        "failures": [{"node": "bad"}], "phase_state_changes": [
            {"node": "good", "after": "x" * 2000},
            {"node": "bad", "after": {"grasped": False}}]}]}
    compact = compact_feedback(feedback, max_characters=1500)
    assert compact["evidence_compacted"]
    assert compact["records"][0]["phase_state_changes"][0]["node"] == "bad"
    assert len(feedback["records"][0]["phase_state_changes"]) == 2


def test_evaluate_and_revise_without_api_or_simulator(harness, monkeypatch):
    payload = task_payload()
    payload["runner"] = {"command": ["fake-worker", "{graph}", "{output}", "{seed}"]}
    harness.task_path.write_text(json.dumps(payload))
    harness = Harness(harness.task_path, harness.session, harness.provider)
    version = harness.generate()

    def run(command, **kwargs):
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert kwargs["stdin"] == subprocess.DEVNULL
        output = Path(command[2])
        output.mkdir()
        graph_path = Path(command[1])
        receipt = receipt_for(read(graph_path), digest(graph_path), int(command[3]))
        (output / "receipt.json").write_text(json.dumps(receipt))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("gap_harness.engine.subprocess.run", run)
    result = harness.evaluate(version)
    assert result["successes"] == 2
    assert (version / "accepted.json").is_file()
    assert (version / "evaluation/seed-0-feedback.json").is_file()
    original = (version / "feedback.json").read_bytes()
    assert harness.evaluate(version)["successes"] == 2
    assert (version / "feedback.json").read_bytes() == original
    assert (version / "evaluation-0002/feedback.json").is_file()


def test_modified_graph_not_executed(harness):
    payload = task_payload()
    payload["runner"] = {"command": ["must-not-run"]}
    harness.task_path.write_text(json.dumps(payload))
    harness = Harness(harness.task_path, harness.session, harness.provider)
    version = harness.generate()
    (version / "graph.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        harness.evaluate(version)


def test_startup_failure_stops_suite_without_blame_or_revision(harness, monkeypatch):
    payload = task_payload()
    payload["runner"] = {"command": ["fake-worker", "{output}"]}
    harness.task_path.write_text(json.dumps(payload))
    harness = Harness(harness.task_path, harness.session, harness.provider)
    version = harness.generate()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr("gap_harness.engine.subprocess.run", run)
    result = harness.evaluate(version)
    assert len(calls) == 1
    assert result["success_rate"] is None
    assert not result["evaluation_complete"]
    assert result["evaluated_trials"] == 0
    assert result["infrastructure_failures"] == 1
    assert result["failures_by_subgraph"] == {}
    assert result["records"][0]["failures"] == []
    assert read(version / "selection.json")["best_version"] is None
    with pytest.raises(ValueError, match="reevaluation"):
        harness.generate()
    assert len(harness.versions()) == 1


def test_libero_worker_gets_prepared_config_and_no_input(harness, monkeypatch):
    payload = task_payload()
    payload["runner"] = {"command": ["python", "-m", "gap_libero.cli"]}
    harness.task_path.write_text(json.dumps(payload))
    harness = Harness(harness.task_path, harness.session, harness.provider)
    version = harness.generate()
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    captured = []

    def run(command, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("gap_harness.engine.subprocess.run", run)
    harness.evaluate(version)
    assert captured[0]["stdin"] == subprocess.DEVNULL
    assert Path(captured[0]["env"]["LIBERO_CONFIG_PATH"]) == (
        captured[0]["cwd"] / "cache/libero-config")


def test_response_transport_uses_strict_schema_and_no_stored_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-not-a-real-key")
    captured = []
    def respond(request):
        captured.append(request)
        return httpx.Response(200, json={"id": "test", "status": "completed", "output": [
            {"type": "reasoning", "summary": []}, {"type": "message", "content": [
                {"type": "output_text", "text": json.dumps(topology_payload())}]}]})
    provider = OpenAIResponses(transport=httpx.MockTransport(respond))
    response = provider.complete("instructions", {"task": "test"}, Topology)
    body = json.loads(captured[0].content)
    assert str(captured[0].url) == "https://api.openai.com/v1/responses"
    assert body["model"] == "gpt-6.1-sol"
    assert body["reasoning"] == {"effort": "medium"}
    assert body["store"] is False
    assert body["text"]["format"]["strict"] is True
    assert Topology.model_validate_json(response["text"])


def test_provider_requires_key_and_redacts_http_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="privately"):
        OpenAIResponses().complete("", {}, Topology)
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-secret")
    transport = httpx.MockTransport(lambda request: httpx.Response(401, text="unit-test-secret"))
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        OpenAIResponses(transport=transport).complete("", {}, Topology)
    assert "unit-test-secret" not in str(error.value)


def test_task_examples_are_loadable_without_inference(tmp_path):
    for filename in ("libero-grocery.json", "inspection-example.json"):
        harness = Harness(ROOT / "harness/tasks" / filename, tmp_path / filename)
        assert harness.task.skills
        assert not harness.session.exists()


def test_different_task_supports_conditional_outcomes():
    task = Task.model_validate_json(
        (ROOT / "harness/tasks/inspection-example.json").read_text())
    topology = Topology.model_validate({"graph_id": "inspection", "rationale": "branching",
        "start": "inspect", "terminals": [{"name": "DONE", "status": "succeeded"},
        {"name": "ABORT", "status": "failed"}], "subgraphs": [
            {"name": name, "skill": skill, "description": "inspection phase", "inputs": [],
             "outputs": [], "exits": exits, "on_error": "error", "routes": routes}
            for name, skill, exits, routes in [
                ("inspect", "inspect-component", ["accepted", "rejected"], [
                    {"outcome": "accepted", "target": "accept"},
                    {"outcome": "rejected", "target": "reject"},
                    {"outcome": "error", "target": "ABORT"}]),
                ("accept", "record-inspection", ["recorded"], [
                    {"outcome": "recorded", "target": "DONE"},
                    {"outcome": "error", "target": "ABORT"}]),
                ("reject", "record-inspection", ["recorded"], [
                    {"outcome": "recorded", "target": "DONE"},
                    {"outcome": "error", "target": "ABORT"}])]]})
    subgraphs = [Subgraph.model_validate({"name": "inspect", "rationale": "classify",
        "start": "capture", "nodes": [
            {"name": "capture", "kind": "tool", "action": "capture_component", "routes": [
                {"outcome": "succeeded", "target": "classify"},
                {"outcome": "failed", "target": "error"}]},
            {"name": "classify", "kind": "tool", "action": "classify_component", "routes": [
                {"outcome": "succeeded", "target": "accepted"},
                {"outcome": "rejected", "target": "rejected"},
                {"outcome": "failed", "target": "error"}]}]})]
    for name in ("accept", "reject"):
        subgraphs.append(Subgraph.model_validate({"name": name, "rationale": "record",
            "start": "record", "nodes": [{"name": "record", "kind": "tool",
                "action": f"record_{name}", "routes": [
                    {"outcome": "succeeded", "target": "recorded"},
                    {"outcome": "failed", "target": "error"}]}]}))
    graph = compile_workflow(topology, subgraphs, task)
    actions = {action.name: (lambda state: NodeResult())
               for skill in task.skills for action in skill.actions}
    actions["classify_component"] = lambda state: NodeResult("rejected")
    result = GraphPolicyExecutor(GraphPolicy.from_dict(graph), actions).execute()
    assert result.status == "succeeded"
    assert result.trace[-1].action == "record_reject"


def test_action_prerequisites_are_checked_on_all_paths():
    payload = task_payload()
    payload["skills"][0]["actions"].append(
        {"name": "prepare", "description": "prepare first"})
    payload["skills"][0]["actions"][0]["requires"] = ["prepare"]
    with pytest.raises(ValueError, match="prerequisites"):
        compile_workflow(Topology.model_validate(topology_payload()),
            [Subgraph.model_validate(subgraph_payload("phase0"))], Task.model_validate(payload))


def test_false_success_receipt_cannot_skip_trace_routes():
    graph = compiled(2)
    receipt = receipt_for(graph, "hash")
    receipt["graph"]["trace"] = receipt["graph"]["trace"][1:]
    trial = build_trial_feedback(receipt, Task.model_validate(task_payload()), graph,
        CheckpointSet.model_validate(checks_payload(2)), "hash", 0)
    assert not trial["known_trace"] and not trial["success"]


def test_structured_output_schemas_require_all_fields():
    def inspect(schema):
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                assert schema["additionalProperties"] is False
                assert set(schema["properties"]) == set(schema["required"])
            for value in schema.values():
                inspect(value)
        elif isinstance(schema, list):
            for value in schema:
                inspect(value)
    for model in (Topology, Subgraph, CheckpointSet):
        inspect(model.model_json_schema())


def test_grocery_catalog_compiles_existing_successful_action_sequence():
    task = Task.model_validate_json((ROOT / "harness/tasks/libero-grocery.json").read_text())
    topology = topology_payload(2)
    subgraphs = []
    for index, skill in enumerate(task.skills):
        name = f"phase{index}"
        topology["subgraphs"][index]["skill"] = skill.name
        nodes = []
        for action_index, action in enumerate(skill.actions):
            nodes.append({"name": f"step{action_index}", "action": action.name,
                "kind": action.kind, "routes": [{"outcome": "succeeded", "target":
                    f"step{action_index+1}" if action_index < len(skill.actions)-1 else "ready"},
                    {"outcome": "failed", "target": "error"}]})
        subgraphs.append(Subgraph.model_validate({"name": name, "rationale": "existing sequence",
                                                 "start": "step0", "nodes": nodes}))
    graph = compile_workflow(Topology.model_validate(topology), subgraphs, task)
    assert len(graph["nodes"]) == 10
    assert [node["action"] for node in graph["nodes"]] == task.required_actions
