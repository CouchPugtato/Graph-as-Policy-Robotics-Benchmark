"""Select official environments for the shared LIBERO runner."""

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .catalog import BENCHMARKS, PRO_DATA_REVISION, ROOT, profile


def tasks(selected):
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[selected["suite"]](task_order_index=0)
    rows = []
    for index in range(suite.n_tasks):
        task = suite.get_task(index)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        states = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        rows.append(
            {
                "task_id": index,
                "name": task.name,
                "language": task.language,
                "bddl": str(bddl),
                "initial_states": str(states),
                "files_present": bddl.is_file() and states.is_file(),
            }
        )
    return rows


def worker(args, selected):
    rows = tasks(selected)
    if args.mode == "list":
        return {
            "ok": True,
            "environment_family": "upstream",
            "tasks": rows,
            "task_count": len(rows),
            "simulator_started": False,
        }
    if args.task_id < 0 or args.task_id >= len(rows):
        raise ValueError("task id is outside the selected upstream suite")
    task = rows[args.task_id]
    if not task["files_present"]:
        raise FileNotFoundError("official task definition or initial states are missing")
    from gap_libero.cli import execute

    args.suite = selected["suite"]
    args.expected_selection = (args.suite, args.task_id, args.init_state_id, args.seed)
    args.capture_initial_observation = True
    result = execute(args)
    hashes = {
        key: hashlib.sha256(Path(task[field]).read_bytes()).hexdigest()
        for key, field in (
            ("task_definition_sha256", "bddl"),
            ("initial_states_sha256", "initial_states"),
        )
    }
    return {
        **result,
        **hashes,
        "simulation_evaluated": args.mode == "run",
        "motion_enabled": args.mode == "run",
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("list", "capture", "run"))
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="libero-object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.mode != "run" and (args.enable_motion or args.graph or args.plan or args.record):
        parser.error("graph motion and recording options require run mode")
    if min(args.task_id, args.init_state_id, args.seed) < 0:
        parser.error("task state and seed must be nonnegative")
    if args.mode == "run" and (not args.enable_motion or not args.graph or not args.plan):
        parser.error("run requires --graph --plan --enable-motion")
    if not args.worker:
        # a short lived worker keeps upstream imports out of the custom runtime
        command = [sys.executable, "-m", "gap_libero.benchmarks.cli", *argv, "--worker"]
        try:
            completed = subprocess.run(command, timeout=180, check=False)
            sys.exit(completed.returncode)
        except subprocess.TimeoutExpired:
            print("Upstream benchmark worker exceeded its 180 second budget", file=sys.stderr)
            sys.exit(1)
    selected = profile(args.benchmark)
    if args.mode != "list":
        args.output = args.output or selected["output_root"] / datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        allowed = (ROOT / "results/libero/upstream", ROOT / "results/benchmarks")
        if not any(args.output.resolve().is_relative_to(path.resolve()) for path in allowed):
            parser.error("upstream outputs must stay under results/libero/upstream")
        args.output.mkdir(parents=True, exist_ok=False)
    try:
        from .runtime import activate

        selected = activate(args.benchmark)
        result = worker(args, selected)
    except Exception as error:
        result = {
            "ok": False,
            "error_type": type(error).__name__,
            "detail": str(error),
            "traceback": traceback.format_exc(),
        }
    result.update(
        environment_family="upstream",
        benchmark=args.benchmark,
        source_revision=selected["revision"],
        suite=selected["suite"],
        source=str(selected["source"]),
        data_revision=(PRO_DATA_REVISION if args.benchmark.startswith("libero-pro-") else None),
    )
    if args.mode != "list":
        (args.output / "receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["output"] = str(args.output.resolve())
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
