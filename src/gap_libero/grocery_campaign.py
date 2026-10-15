"""Sequential bounded trials of the unchanged graph on the matched grocery layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "graphs/generated/codex-20261009T145742Z-f28cf576/accepted.json"
ISAAC_CAMPAIGN = ROOT / "results/cylinder-center-v24/ten-seed-20261009T164626391241Z"


def run_campaign(output: Path, graph: Path = DEFAULT_GRAPH, seeds=range(10)) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(graph.read_bytes()).hexdigest()
    records = []
    for seed in seeds:
        verified = False
        destination = output / f"seed-{seed}"
        command = [sys.executable, "-m", "gap_libero.cli", "compare-grocery",
                   "--seed", str(seed), "--graph", str(graph), "--enable-motion",
                   "--output", str(destination)]
        print(f"Running matched grocery seed {seed}", flush=True)
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=90,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
            (output / f"seed-{seed}.log").write_text(process.stdout + process.stderr,
                                                    encoding="utf-8")
            receipt_path = destination / "receipt.json"
            receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
            trace = receipt.get("graph", {}).get("trace", [])
            verified = (process.returncode == 0 and receipt.get("ok") is True
                        and receipt.get("graph_sha256") == digest
                        and receipt.get("graph_modified") is False
                        and len(trace) == 10
                        and all(item["outcome"] == "succeeded" for item in trace)
                        and all(receipt.get("evaluation", {}).get("checkpoints", {}).values()))
            record = {"seed": seed, "ok": verified, "receipt": str(receipt_path),
                      "implementation_files_sha256": receipt.get("implementation_files_sha256"),
                      "last_node": trace[-1]["node_id"] if trace else None,
                      "detail": receipt.get("detail") or (
                          trace[-1]["detail"] if trace else "no trace")}
        except subprocess.TimeoutExpired:
            record = {"seed": seed, "ok": False, "detail": "worker exceeded ninety second limit"}
        baseline = ISAAC_CAMPAIGN / f"seed-{seed}" / "trial.json"
        if baseline.is_file():
            trial = json.loads(baseline.read_text())
            if trial.get("graph_sha256") == digest:
                record["historical_isaac_status"] = trial["status"]
                record["historical_isaac_trial"] = str(baseline)
        records.append(record)
        print(f"Seed {seed}: {'succeeded' if verified else 'failed'}", flush=True)
    summary = {"backend": "libero", "custom_task": "grocery_ground",
               "graph": str(graph), "graph_sha256": digest, "graph_modified": False,
               "trials": len(records), "successes": sum(item["ok"] for item in records),
               "records": records,
               "historical_isaac_trials": sum(
                   "historical_isaac_status" in item for item in records),
               "historical_isaac_successes": sum(
                   item.get("historical_isaac_status") == "succeeded" for item in records),
               "historical_isaac_campaign": str(ISAAC_CAMPAIGN),
               "limits": ["oracle geometry LIBERO versus historical perceived geometry Isaac",
                          "custom LIBERO task not an official suite benchmark",
                          "controller and physics differ so this is not a simulator ranking"]}
    implementations = [record.get("implementation_files_sha256") for record in records]
    summary["consistent_implementation"] = bool(implementations) and all(
        value and value == implementations[0] for value in implementations)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"successes": summary["successes"], "trials": len(records),
                      "summary": str(output / "summary.json")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--seeds", nargs="+", type=int, choices=range(10), default=list(range(10)))
    args = parser.parse_args()
    run_campaign(args.output, args.graph, args.seeds)


if __name__ == "__main__":
    main()
