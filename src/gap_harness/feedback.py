"""Deterministic evidence packets for the rehearsal and refinement loop."""

from __future__ import annotations

import json
from collections import Counter

from gap_harness.models import CheckpointSet, Task
from gap_harness.workflow import lookup


def trace_matches_graph(receipt, graph):
    nodes = {node["id"]: node for node in graph["nodes"]}
    trace = receipt.get("graph", {}).get("trace", [])
    current = graph["start"]
    if not isinstance(trace, list) or not trace or len(trace) > 64:
        return False
    for sequence, event in enumerate(trace):
        node = nodes.get(current)
        if node is None or not isinstance(event, dict):
            return False
        if (event.get("sequence") != sequence or event.get("node_id") != current
                or event.get("action") != node["action"] or event.get("skill") != node["skill"]):
            return False
        next_node = node["routes"].get(event.get("outcome"))
        if next_node is None or next_node != event.get("next_node"):
            return False
        current = next_node
    result = receipt.get("graph", {})
    return (result.get("graph_id") == graph["graph_id"] and result.get("terminal") == current
            and graph["terminals"].get(current) == result.get("status"))


def build_trial_feedback(receipt, task: Task, graph, checkpoints: CheckpointSet,
                         graph_sha256, seed, *, process_ok=True):
    trace = receipt.get("graph", {}).get("trace", [])
    membership = graph["metadata"]["subgraph_membership"]
    graph_bound = receipt.get("graph_sha256") == graph_sha256
    seed_bound = type(receipt.get("seed")) is int and receipt["seed"] == seed
    known_trace = trace_matches_graph(receipt, graph)
    evaluated = graph_bound and seed_bound and known_trace
    criteria = {path: lookup(receipt, path) is True for path in task.required_true}
    if not isinstance(trace, list):
        trace = []
    kinds = graph.get("metadata", {}).get("node_kinds", {})
    failed = [event for event in trace if isinstance(event, dict)
              and event.get("outcome") != (
                  "true" if kinds.get(event.get("node_id")) == "router" else "succeeded")]
    attribution = []
    for event in failed:
        attribution.append({"node": event.get("node_id"),
                            "subgraph": membership.get(event.get("node_id")),
                            "action": event.get("action"), "detail": event.get("detail", "")})
    checked = []
    for check in checkpoints.checkpoints:
        value = lookup(receipt, check.evidence)
        expected = {"true": True, "false": False,
                    "succeeded": "succeeded", "failed": "failed"}[check.expected]
        checked.append({**check.model_dump(), "observed": value,
                        "passed": type(value) is type(expected) and value == expected})
        if evaluated and value is not None and not checked[-1]["passed"]:
            attribution.append({"node": None, "subgraph": check.subgraph,
                "action": None, "detail": f"postcondition {check.name} failed", "kind":
                "checkpoint", "evidence": check.evidence, "observed": value})
    # generated checkpoints can add diagnostics but cannot replace fixed task criteria
    success = (process_ok and graph_bound and seed_bound and known_trace
               and receipt.get("ok") is True
               and receipt.get("graph", {}).get("status") == "succeeded"
               and all(criteria.values()) and all(check["passed"] for check in checked))
    trajectory = receipt.get("trajectory", [])
    before = receipt.get("initial_state")
    phases = []
    for index, sample in enumerate(trajectory):
        after = sample.get("after", sample)
        event = trace[index] if index < len(trace) and isinstance(trace[index], dict) else {}
        phases.append({"action": sample.get("action"),
                       "node": event.get("node_id"),
                       "subgraph": membership.get(event.get("node_id")),
                       "before": sample.get("before", before), "after": after})
        before = after
    return {"seed": seed, "success": bool(success), "process_ok": process_ok,
            "evaluated": evaluated,
            "failure_kind": None if success else ("policy" if evaluated else "infrastructure"),
            "graph_bound": graph_bound, "seed_bound": seed_bound, "known_trace": known_trace,
            "fixed_criteria": criteria, "checkpoints": checked,
            "failures": attribution if evaluated else [],
            "detail": receipt.get("detail", ""), "trace": trace,
            "phase_state_changes": phases,
            "contacts": receipt.get("evaluation", {}).get("disallowed_collisions", []),
            "evidence_limits": ([] if phases and phases[0]["before"] is not None else
                ["runner did not supply an initial state for the first phase"])}


def aggregate_feedback(trials):
    evaluated = sum(trial["evaluated"] for trial in trials)
    counts = Counter(failure["subgraph"] or "unattributed"
                     for trial in trials for failure in trial["failures"])
    return {"trials": len(trials), "successes": sum(trial["success"] for trial in trials),
            "evaluated_trials": evaluated, "infrastructure_failures": len(trials) - evaluated,
            "evaluation_complete": evaluated == len(trials),
            "success_rate": (sum(trial["success"] for trial in trials) / len(trials)
                             if evaluated == len(trials) else None),
            "failures_by_subgraph": dict(counts), "records": trials,
            "revision_guidance": (["repair runner infrastructure and reevaluate this graph "
                                   "before requesting a policy revision"]
                                  if evaluated != len(trials) else [
                "compare state and contact changes around the first failing phase",
                "separate perception geometry control and verification failures",
                "repair responsible subgraphs and preserve successful behavior",
                "do not change task criteria physics bindings or seed specific coordinates",
                "rerun the complete configured seed suite to detect regressions"])}


def compact_feedback(feedback, max_characters=90000):
    if len(json.dumps(feedback)) <= max_characters:
        return feedback
    records = []
    for trial in feedback.get("records", []):
        failures = trial.get("failures", [])
        phases = trial.get("phase_state_changes", [])
        failed_nodes = {failure.get("node") for failure in failures if failure.get("node")}
        selected = [phase for phase in phases if phase.get("node") in failed_nodes][:1]
        if not selected:
            selected = phases[-1:]
        records.append({key: trial.get(key) for key in (
            "seed", "success", "process_ok", "graph_bound", "seed_bound", "known_trace",
            "evaluated", "failure_kind",
            "fixed_criteria", "checkpoints", "failures", "detail", "contacts")})
        records[-1]["phase_state_changes"] = selected
        records[-1]["trace"] = [{key: event.get(key) for key in (
            "node_id", "action", "outcome", "detail")} for event in trial.get("trace", [])]
    summary = {key: value for key, value in feedback.items() if key != "records"}
    summary.update(records=records, evidence_compacted=True,
                   full_evidence="retained in the parent version feedback and evaluation files")
    if len(json.dumps(summary)) > max_characters:
        for record in records:
            record.pop("phase_state_changes", None)
            record.pop("trace", None)
            record.pop("contacts", None)
        summary["state_evidence_omitted"] = "packet size cap inspect saved raw evidence locally"
    return summary
