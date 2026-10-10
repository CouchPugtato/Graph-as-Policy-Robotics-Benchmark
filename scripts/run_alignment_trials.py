"""Explicit revision selection; bounded seed-9 experiments with all attempts retained."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_layout_trials import ROOT, command

sys.path.insert(0, str(ROOT / "src"))
from gap_isaac.bounded_isaac import stop_owned_tree

INPUT = ROOT / "results/cylinder-center-v24/ten-seed-20261009T164626391241Z/seed-9/trial.json"
ENTRY_POINTS = {
    "diagnostic": {"motion": "test-alignment-diagnostic.ps1",
                   "revision": "alignment-endpoint-v24-diagnostic-v1"},
    "v25": {"motion": "test-graph-pick-place-v25.ps1",
            "feasibility": "test-perceived-feasibility-v25.ps1",
            "revision": "graph-ground-perceived-pick-place-v25-alignment",
            "geometry_adapter": "gap_isaac.perceived_feasibility_v24",
            "center_fit": "gap_isaac.cylinder_center"},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_preserved():
    # historical comparisons require the original source files to stay unchanged
    for item in json.loads((ROOT / "revisions/v24-frozen/manifest.json").read_text())["files"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Preserved v24 file changed: {item['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", choices=ENTRY_POINTS, required=True)
    parser.add_argument("--seed", type=int, choices=[9], required=True)
    args = parser.parse_args()
    assert_preserved()
    entry = ENTRY_POINTS[args.revision]
    sources = [p for folder in ("src", "isaac", "scripts", "configs", "graphs")
               for p in (ROOT / folder).rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and
               p.suffix in (".py", ".ps1", ".json", ".yaml", ".toml", ".md")]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    directory = ROOT / "results/alignment-v25" / (
        args.revision + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(parents=True, exist_ok=False)
    source = json.loads(INPUT.read_text())
    state = {"seed": 9, "revision": entry["revision"], "entry_points": entry,
             "source_trial": str(INPUT), "source_hashes": hashes, "steps": [],
             "status": "running", "infrastructure_retries": 0}

    def save():
        (directory / "attempt.json").write_text(json.dumps(state, indent=2))

    def check_sources():
        assert_preserved()
        for path, sha in hashes.items():
            if digest(ROOT / path) != sha:
                raise RuntimeError(f"Frozen experiment source changed: {path}")

    def run(name, script, arguments, timeout=120):
        check_sources()
        log_path = directory / (name + ".log")
        with log_path.open("x") as log:
            process = subprocess.Popen(command(script, arguments), cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                if process.poll() is None:
                    stop_owned_tree(process)
        text = log_path.read_text(errors="replace")
        matches = re.findall(r"Saved (?:graph result|diagnostic): (.+\.json)", text)
        result = matches[-1].strip() if matches else None
        step = {"name": name, "script": script, "log": str(log_path),
                "owned_pid": process.pid, "exit_code": process.returncode,
                "timed_out": timed_out, "result": result}
        state["steps"].append(step)
        save()
        check_sources()
        print(json.dumps(step), flush=True)
        receipt = json.loads(Path(result).read_text()) if result else {}
        if process.returncode or timed_out or not receipt.get("ok"):
            if (name == "feasibility" and receipt.get("error_type") == "DiagnosticTimeout"
                    and receipt.get("commands_applied") == 0):
                state["infrastructure_retries"] += 1
                save()
                return run("feasibility-retry", script, arguments, timeout)
            raise RuntimeError(f"{name} failed; attempt retained")
        if receipt.get("run_revision") != entry["revision"]:
            raise RuntimeError(f"{name} receipt revision mismatch")
        return result

    save()
    print(f"Attempt evidence: {directory}", flush=True)
    geometry = ["-Stage", source["artifacts"]["stage"], "-TargetGeometry",
                source["artifacts"]["cylinder_geometry"], "-BasketGeometry",
                source["artifacts"]["basket_geometry"]]
    try:
        if args.revision == "diagnostic":
            run("alignment", entry["motion"],
                [*geometry, "-Seed", 9, "-Headless", "-Mode", "Grasp"])
            state["status"] = "diagnostic_completed"
        else:
            run("feasibility", entry["feasibility"], geometry, 110)
            grasp = run("grasp", entry["motion"],
                [*geometry, "-Seed", 9, "-Headless", "-Mode", "Grasp"])
            graph = run("graph", entry["motion"],
                [*geometry, "-Seed", 9, "-Headless", "-Mode", "Graph",
                "-GraspResult", grasp, "-Graph", source["graph"]])
            receipt = json.loads(Path(graph).read_text())
            assert receipt["graph_sha256"] == source["graph_sha256"]
            assert len(receipt["graph"]["trace"]) == 10
            assert all(t["outcome"] == "succeeded" for t in receipt["graph"]["trace"])
            assert all(receipt["evaluation"]["checkpoints"].values())
            assert receipt["evaluation"]["disallowed_collisions"] == []
            state["status"] = "succeeded"
    except Exception as error:
        state.update(status="failed", error=str(error))
    finally:
        check_sources()
        save()
    print(json.dumps({"status": state["status"],
                      "attempt": str(directory / "attempt.json")}), flush=True)
    return 0 if state["status"] in ("succeeded", "diagnostic_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
