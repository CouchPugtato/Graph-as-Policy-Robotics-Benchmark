"""Validate an iteration, freeze its source hashes, and evaluate seeds 0–9 serially."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_layout_trials import ROOT, Trial

from gap_isaac.policy_generation import (
    MODEL,
    accept,
    load_json,
    validate_generated_graph,
    verify_accepted,
    verify_package,
)


def source_hashes(root):
    # record the code used for this campaign before any trials begin
    files = set()
    for folder, pattern in (
        ("src", "*.py"),
        ("isaac", "*.py"),
        ("scripts", "*.py"),
        ("scripts/windows", "*.ps1"),
        ("configs", "*.yaml"),
    ):
        files.update((root / folder).rglob(pattern))
    files.add(root / "pyproject.toml")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }


def revision(root):
    tree = ast.parse((root / "isaac/validate_graph_pick_place.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RUN_REVISION" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Controller must declare RUN_REVISION")


def run_tests(directory):
    with (directory / "tests.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    if result.returncode:
        raise ValueError(f"Local tests failed; see {directory / 'tests.log'}")


def check_resources(directory, seed):
    python = ROOT / ".venv-perception/Scripts/python.exe"
    if not python.is_file() or not Path("C:/isaacsim/python.bat").is_file():
        raise ValueError("Native perception environment and C:/isaacsim/python.bat are required")
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "from gap_isaac.dino_smoke import check_stopped_apps; "
        "from gap_isaac.cuda_probe import check_idle_gpu; "
        "check_stopped_apps(); print(check_idle_gpu())"
    )
    with (directory / f"resources-{seed}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [str(python), "-c", code], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=30
        )
    if result.returncode:
        raise ValueError("Resource check failed; close Isaac/LM Studio and inspect resource log")


def validate_result(result, expected_revision, graph_hash, seed):
    evaluation = result["evaluation"]
    checkpoints = evaluation["checkpoints"]
    graph = result["graph"]
    if (
        result.get("ok") is not True
        or result.get("run_revision") != expected_revision
        or result.get("graph_sha256") != graph_hash
        or evaluation.get("seed") != seed
        or evaluation.get("status") != "succeeded"
        or set(checkpoints)
        != {"target_grasped", "target_lifted", "target_released", "target_contained"}
        or any(value is not True for value in checkpoints.values())
        or evaluation.get("disallowed_collisions") != []
        or graph.get("status") != "succeeded"
        or graph.get("terminal") != "SUCCEEDED"
        or len(graph.get("trace", [])) != 10
        or any(node.get("outcome") != "succeeded" for node in graph["trace"])
    ):
        raise ValueError("Graph/evaluator receipt failed independent campaign validation")


def write_summary(directory, summary):
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = summary["seeds"]
    passed = sum(row["status"] == "succeeded" for row in rows)
    lines = [
        "# Iteration evaluation",
        "",
        f"Status: {summary['status']}",
        (
            f"End-to-end successes: {passed}/10; completed seeds: {len(rows)}/10."
            if rows
            else "No simulation trials performed."
        ),
        "",
        f"Controller revision: `{summary.get('run_revision', 'not validated')}`",
        f"Graph: `{summary.get('graph', 'not accepted')}`",
        "",
        "| Seed | Outcome | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['status']} | [Trial](trials/seed-{row['seed']}/trial.json) |"
        )
    lines += [
        "",
        "No automatic retries. Failed prerequisites are included in the denominator.",
        "Source hashes are recorded; they do not certify the safety of code changes.",
    ]
    if summary.get("error"):
        lines += ["", "Error: " + summary["error"]]
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--session-note", help="Required only when accepting a new candidate")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    directory = (
        ROOT / "results/iterations" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    directory.mkdir(parents=True)
    summary = {"status": "validating", "seeds": [], "automatic_retries": 0}
    print(f"Iteration evidence: {directory}", flush=True)
    try:
        bundle = args.bundle.resolve()
        verify_package(bundle)
        validate_generated_graph(load_json(bundle / "candidate.json"))
        graph = bundle / "accepted.json"
        if graph.exists():
            verify_accepted(graph)
        elif not args.session_note:
            raise ValueError("New candidate requires --session-note for declared model provenance")
        current_revision = revision(ROOT)
        if current_revision != args.expected_revision:
            raise ValueError(
                f"Expected revision {args.expected_revision}; found {current_revision}"
            )
        print("Validating graph and running local tests; no simulator or models.", flush=True)
        tested_sources = source_hashes(ROOT)
        run_tests(directory)
        if source_hashes(ROOT) != tested_sources:
            raise ValueError("Source/configuration changed while tests ran")
        if not graph.exists():
            accept(bundle, MODEL, args.session_note)
        receipt = verify_accepted(graph)
        frozen = source_hashes(ROOT)
        summary.update(
            graph=str(graph),
            graph_sha256=receipt["graph_sha256"],
            run_revision=current_revision,
            source_sha256=frozen,
        )
        if args.validate_only:
            summary["status"] = "validated-not-run"
            return 0
        trials = directory / "trials"
        trials.mkdir()
        for seed in range(10):
            if source_hashes(ROOT) != frozen:
                raise ValueError("Source/configuration changed during campaign; stopping motion")
            check_resources(directory, seed)
            trial = Trial(seed, trials, graph)
            try:
                trial.execute()
                result = load_json(Path(trial.state["artifacts"]["graph"]))
                validate_result(result, current_revision, receipt["graph_sha256"], seed)
            except Exception as error:
                trial.state.update(status="failed", error=f"{type(error).__name__}: {error}")
                trial.save()
                print(f"seed {seed}: FAILED: {error}", flush=True)
            stage = Path(trial.state["artifacts"].get("stage", "missing.usd"))
            summary["seeds"].append(
                {
                    "seed": seed,
                    "status": trial.state["status"],
                    "trial": str(trial.directory / "trial.json"),
                    "artifacts": trial.state["artifacts"],
                    "error": trial.state.get("error"),
                    "stage_sha256": hashlib.sha256(stage.read_bytes()).hexdigest()
                    if stage.is_file()
                    else None,
                }
            )
            if source_hashes(ROOT) != frozen:
                raise ValueError("Source/configuration changed during trial; campaign invalid")
            summary["status"] = "running"
            write_summary(directory, summary)
        summary["status"] = "complete"
        return int(any(row["status"] != "succeeded" for row in summary["seeds"]))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        summary.update(status="blocked", error=f"{type(error).__name__}: {error}")
        return 2
    except KeyboardInterrupt:
        summary.update(status="interrupted", error="Operator interrupted campaign")
        return 130
    finally:
        write_summary(directory, summary)
        print(f"Saved summary: {directory / 'summary.md'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
