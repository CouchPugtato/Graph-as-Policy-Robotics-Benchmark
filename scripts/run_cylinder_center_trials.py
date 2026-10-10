"""Sequential bounded v24 evaluation; original v23 runner and results stay intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_layout_trials import ROOT, Trial, command, payload

sys.path.insert(0, str(ROOT / "src"))
from gap_isaac.bounded_isaac import stop_owned_tree
from gap_isaac.policy_generation import verify_accepted

GRAPH = ROOT / "graphs/generated/codex-20261009T145742Z-f28cf576/accepted.json"
BASE_INPUTS = ROOT / "results/layout-trials/20261009T150043963953Z"
REVISION = "graph-ground-perceived-pick-place-v24-cylinder-center"


def assert_frozen():
    manifest_path = ROOT / "revisions/v23-frozen/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for item in manifest["files"]:
        if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Frozen v23 input changed: {item['path']}")


class CylinderTrial(Trial):
    def __init__(self, seed, directory, *, baseline=False):
        self.baseline = baseline
        super().__init__(seed, directory, GRAPH)
        self.state.update(run_revision=("graph-ground-perceived-pick-place-v23-diagnostics"
                                        if baseline else REVISION),
                          infrastructure_retries=0)
        self.save()

    def run(self, name, script, arguments, timeout=120, allow_failed_result=False):
        if script == "test-graph-pick-place.ps1":
            script = "test-graph-pick-place-v24.ps1"
            if self.baseline:
                arguments = [*arguments, "-BaselineDiagnostics"]
        if script == "test-perceived-feasibility.ps1" and not self.baseline:
            script = "test-perceived-feasibility-v24.ps1"
        print(f"{self.directory}: {name} starting", flush=True)
        log_path = self.directory / f"{name}.log"
        started = time.monotonic()
        timed_out = False
        interrupted = False
        cleanup_error = None
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command(script, arguments), cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            except KeyboardInterrupt:
                interrupted = True
            finally:
                if process.poll() is None:
                    try:
                        stop_owned_tree(process)
                    except (OSError, subprocess.SubprocessError) as error:
                        cleanup_error = str(error)
        output = log_path.read_text(encoding="utf-8", errors="replace")
        entry = {"name": name, "script": script, "exit_code": process.returncode,
                 "wall_s": round(time.monotonic() - started, 3), "log": str(log_path),
                 "owned_pid": process.pid, "timed_out": timed_out,
                 "interrupted": interrupted, "cleanup_error": cleanup_error}
        diagnostic = re.findall(r"Saved diagnostic: (.+\.json)", output)
        if diagnostic:
            entry["result"] = diagnostic[-1].strip()
            self.remember(name, entry["result"])
        result_paths = re.findall(r"Saved graph result: (.+\.json)", output)
        if result_paths:
            entry["result"] = result_paths[-1].strip()
            self.remember(name, entry["result"])
        self.state["steps"].append(entry)
        self.save()
        if interrupted or cleanup_error:
            raise RuntimeError(f"Owned-process interruption/cleanup failure: {entry}")
        # One predefined infrastructure-only retry. The original log/result/step stays recorded.
        if name == "feasibility" and diagnostic and process.returncode:
            receipt = json.loads(Path(entry["result"]).read_text(encoding="utf-8"))
            if (receipt.get("error_type") == "DiagnosticTimeout"
                    and receipt.get("commands_applied") == 0):
                self.state["infrastructure_retries"] += 1
                self.save()
                return self.run("feasibility-retry", script, arguments, timeout,
                                allow_failed_result)
        if timed_out or (process.returncode and not allow_failed_result):
            raise RuntimeError(f"{name} failed; attempt retained: {entry}")
        print(f"{self.directory}: {name} exited {process.returncode}", flush=True)
        return output

    def execute_motion(self, stage, geometries):
        if not self.baseline:
            original = geometries["alphabet_soup"]
            fit_path = self.directory / "cylinder-geometry.json"
            output = self.run("cylinder-center", "test-cylinder-center.ps1",
                              ["-TargetGeometry", original,
                               "-BasketGeometry", geometries["basket"], "-Output", fit_path], 30)
            fit_result = self.remember("cylinder_geometry", payload(output, "result"))
            self.state["raw_target_geometry"] = str(original)
            self.save()
            geometries = {**geometries, "alphabet_soup": fit_result}
        super().execute_motion(stage, geometries)
        graph = json.loads(Path(self.state["artifacts"]["graph"]).read_text())
        if graph["run_revision"] != self.state["run_revision"]:
            raise RuntimeError("Unexpected execution revision in successful receipt")


def execute_trial(seed, directory, *, baseline=False, saved=False):
    assert_frozen()
    directory.mkdir(parents=True, exist_ok=True)
    trial = CylinderTrial(seed, directory, baseline=baseline)
    if saved:
        source_path = BASE_INPUTS / f"seed-{seed}/trial.json"
        source = json.loads(source_path.read_text())
        trial.state["source_trial"] = str(source_path)
        trial.state["artifacts"] = {key: value for key, value in source["artifacts"].items()
                                    if key not in ("feasibility", "grasp", "graph", "pair")}
        trial.save()
    try:
        if saved:
            geometries = {name: source["artifacts"][name + "_geometry"]
                          for name in ("alphabet_soup", "basket")}
            output = trial.run("pair", "test-geometry-pair.ps1",
                               ["-Target", geometries["alphabet_soup"],
                                "-Basket", geometries["basket"]], 30)
            trial.remember("pair", payload(output, "result"))
            trial.execute_motion(Path(source["artifacts"]["stage"]), geometries)
        else:
            trial.execute()
    except Exception as error:
        trial.state.update(status="failed", error=f"{type(error).__name__}: {error}")
        trial.save()
        print(f"seed {seed}: FAILED: {error}", flush=True)
    assert_frozen()
    return str(trial.directory / "trial.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paired", "ten-seed"), required=True)
    args = parser.parse_args()
    assert_frozen()
    receipt = verify_accepted(GRAPH)
    directory = ROOT / "results/cylinder-center-v24" / (
        args.mode + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / path for path in (
        "src/gap_isaac/cylinder_center.py", "src/gap_isaac/perceived_feasibility_v24.py",
        "src/gap_isaac/diagnostic_progress_v24.py",
        "isaac/validate_graph_pick_place_v24.py", "isaac/validate_perceived_feasibility_v24.py",
        "scripts/windows/test-graph-pick-place-v24.ps1",
        "scripts/windows/test-perceived-feasibility-v24.ps1",
        "scripts/windows/test-cylinder-center.ps1", "scripts/run_cylinder_center_trials.py")]
    campaign = {"mode": args.mode, "revision": REVISION, "graph": str(GRAPH),
                "graph_sha256": receipt["graph_sha256"], "worker_count": 1,
                "gpu": "RTX 5080", "trials": [],
                "source_hashes": {str(p.relative_to(ROOT)):
                                  hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources}}
    def save():
        (directory / "campaign.json").write_text(json.dumps(campaign, indent=2))
    save()
    print(f"Campaign evidence: {directory}", flush=True)
    if args.mode == "paired":
        selection = [(3, 1), (1, 1), (5, 1), (5, 2), (5, 3)]
        for seed, attempt in selection:
            for baseline in (True, False):
                label = "baseline" if baseline else "candidate"
                path = execute_trial(seed, directory / f"seed-{seed}-pair-{attempt}-{label}",
                                     baseline=baseline, saved=True)
                campaign["trials"].append(path)
                save()
    else:
        for seed in range(10):
            campaign["trials"].append(execute_trial(seed, directory, saved=False))
            save()
    records = [json.loads(Path(p).read_text()) for p in campaign["trials"]]
    campaign["completed"] = True
    campaign["succeeded"] = sum(r["status"] == "succeeded" for r in records)
    campaign["attempt_count"] = len(records)
    campaign["infrastructure_retry_count"] = sum(r["infrastructure_retries"] for r in records)
    save()
    print(json.dumps({key: campaign[key] for key in
                      ("completed", "succeeded", "attempt_count", "infrastructure_retry_count")}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
