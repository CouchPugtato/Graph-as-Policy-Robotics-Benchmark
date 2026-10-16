"""Rehearse reference policies through the real harness without generation calls."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from gap_harness.engine import Harness, digest, dump
from gap_libero.scenario_catalog import reference_workflow
from gap_libero.scenario_specs import SCENARIOS

ROOT = Path(__file__).resolve().parents[2]


def prepare_reference(scenario, output, seeds=None):
    task_path = ROOT / f"harness/tasks/libero-{scenario}-local-qwen.json"
    if seeds is not None:
        task = json.loads(task_path.read_text())
        task["seeds"] = seeds
        task["runner"]["cwd"] = str(ROOT)
        task_path = output / f"{scenario}-task.json"
        dump(task_path, task)
    harness = Harness(task_path, output / scenario)
    harness.prepare()
    version = harness.session / "v0001"
    version.mkdir()
    bundle = reference_workflow(scenario)
    dump(version / "graph.json", bundle.pop("graph"))
    dump(version / "workflow.json", bundle)
    dump(version / "topology.json", bundle["topology"])
    dump(version / "checkpoints.json", {"checkpoints": bundle["checkpoints"]})
    dump(
        version / "lineage.json",
        {
            "parent": None,
            "task_sha256": harness.task_sha256,
            "model": "hand-authored-reference",
            "model_generated": False,
        },
    )
    dump(
        version / "validation.json",
        {
            "ok": True,
            "graph_sha256": digest(version / "graph.json"),
            "execution_validated": False,
            "reference_policy": True,
            "skill_agent_count": 0,
        },
    )
    return harness, version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", choices=range(10))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.seeds is not None and len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be unique")
    output = (
        args.output
        or ROOT
        / "results/scenario-validation"
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "model_generation_calls": 0,
        "local_qwen_required": not args.check_only,
        "check_only": args.check_only,
        "scenarios": {},
        "ready_for_generation": False,
    }
    for scenario in SCENARIOS if args.scenario == "all" else (args.scenario,):
        print(f"Preparing {scenario} reference policy", flush=True)
        harness, version = prepare_reference(scenario, output, args.seeds)
        result = {"catalog_valid": True, "reference_compiles": True, "version": str(version)}
        if not args.check_only:
            feedback = harness.evaluate(version)
            result.update(
                {
                    key: feedback[key]
                    for key in (
                        "trials",
                        "successes",
                        "evaluation_complete",
                        "infrastructure_failures",
                    )
                }
            )
            result["policy_failures"] = [
                {
                    "seed": trial["seed"],
                    "failures": trial["failures"],
                    "missing_executed_actions": trial["missing_executed_actions"],
                    "scenario_evidence": trial["scenario_evidence"],
                }
                for trial in feedback["records"]
                if not trial["success"]
            ]
        summary["scenarios"][scenario] = result
    summary["ready_for_generation"] = not args.check_only and all(
        result["evaluation_complete"] and result["successes"] > 0
        for result in summary["scenarios"].values()
    )
    summary["readiness_definition"] = (
        "catalog compiles and every configured trial supplies valid feedback with at least "
        "one successful reference episode per scenario not a requirement for perfect success"
    )
    dump(output / "readiness.json", summary)
    print(
        json.dumps(
            {
                "output": str(output),
                "ready_for_generation": summary["ready_for_generation"],
                "scenarios": {
                    name: {key: value for key, value in result.items() if key != "policy_failures"}
                    for name, result in summary["scenarios"].items()
                },
            },
            indent=2,
        )
    )
    if not args.check_only and not summary["ready_for_generation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
