"""Run saved-layout trials serially, retaining each stage's artifacts and logs."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gap_isaac.bounded_isaac import powershell_literal, stop_owned_tree  # noqa: E402
from gap_isaac.policy_generation import verify_accepted  # noqa: E402

GRAPH = ROOT / "graphs/generated/codex-20261009T052250Z-d8582280/accepted.json"


def command(script, arguments):
    invocation = "& " + powershell_literal(str(ROOT / "scripts/windows" / script))
    for argument in arguments:
        value = str(argument)
        invocation += " " + (
            value if re.fullmatch(r"-[A-Za-z][A-Za-z0-9]*", value) else powershell_literal(value)
        )
    return [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-WindowStyle",
        "Hidden",
        "-Command",
        "$ErrorActionPreference = 'Stop'; " + invocation + "; exit $LASTEXITCODE",
    ]


def payload(text, key):
    decoder = json.JSONDecoder()
    found = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("ok") is True and key in value:
            found.append(value)
    if not found:
        raise RuntimeError(f"No successful {key} receipt in stage output")
    return found[-1][key]


class Trial:
    def __init__(self, seed, directory, graph=GRAPH):
        self.seed = seed
        self.graph = Path(graph).resolve()
        self.directory = directory / f"seed-{seed}"
        self.directory.mkdir()
        self.state = {
            "seed": seed,
            "status": "running",
            "artifacts": {},
            "steps": [],
            "graph": str(self.graph),
            "graph_sha256": verify_accepted(self.graph)["graph_sha256"],
        }
        self.save()

    def save(self):
        (self.directory / "trial.json").write_text(
            json.dumps(self.state, indent=2), encoding="utf-8"
        )

    def run(self, name, script, arguments, timeout=120, allow_failed_result=False):
        print(f"seed {self.seed}: {name} starting", flush=True)
        log_path = self.directory / f"{name}.log"
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command(script, arguments),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                process.wait(timeout=timeout)
            finally:
                # clean up only the process tree created for this trial stage
                if process.poll() is None:
                    stop_owned_tree(process)
        output = log_path.read_text(encoding="utf-8", errors="replace")
        self.state["steps"].append(
            {
                "name": name,
                "exit_code": process.returncode,
                "wall_s": round(time.monotonic() - started, 3),
                "log": str(log_path),
            }
        )
        self.save()
        if process.returncode and not allow_failed_result:
            raise RuntimeError(f"{name} failed (exit {process.returncode}); see {log_path}")
        print(f"seed {self.seed}: {name} exited {process.returncode}", flush=True)
        return output

    def remember(self, key, value):
        self.state["artifacts"][key] = str(value)
        self.save()
        return str(value)

    def capture(self, stage):
        print(f"seed {self.seed}: capture starting", flush=True)
        environment = os.environ.copy()
        # each capture process gets its own temporary connector credential
        environment["GAP_ISAAC_CONNECTOR_TOKEN"] = secrets.token_hex(32)
        headers = {"Authorization": "Bearer " + environment["GAP_ISAAC_CONNECTOR_TOKEN"]}
        opener = build_opener(ProxyHandler({}))

        def api(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            request = Request(
                "http://127.0.0.1:8765" + path,
                data=data,
                headers={**headers, "Content-Type": "application/json"},
            )
            with opener.open(request, timeout=3) as response:
                return json.load(response)

        log_path = self.directory / "connector.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command(
                    "run-read-only-connector.ps1",
                    ["-Stage", stage, "-Headless", "-CaptureEvery", 12],
                ),
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Connector exited before capture: {log_path}")
                    try:
                        observation = api("/v1/observations/latest")
                        if observation.get("sequence", -1) >= 20:
                            break
                    except URLError:
                        pass
                    time.sleep(0.5)
                else:
                    raise TimeoutError(
                        "Connector did not produce a settled observation within 60 s"
                    )
                capture_log = self.directory / "capture.log"
                with capture_log.open("w", encoding="utf-8") as output:
                    capture = subprocess.Popen(
                        command("capture-observation.ps1", []),
                        cwd=ROOT,
                        env=environment,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    try:
                        capture.wait(timeout=20)
                    finally:
                        if capture.poll() is None:
                            stop_owned_tree(capture)
                if capture.returncode:
                    raise RuntimeError(f"Observation capture failed: {capture_log}")
                bundle = payload(
                    capture_log.read_text(encoding="utf-8", errors="replace"), "bundle"
                )
                self.remember("observation", bundle)
                api("/v1/commands/stop", {"command_id": f"seed-{self.seed}-capture-complete"})
                process.wait(timeout=20)
                if process.returncode:
                    raise RuntimeError(f"Connector shutdown failed: {log_path}")
                return str(Path(bundle) / "rgb.png")
            finally:
                if process.poll() is None:
                    stop_owned_tree(process)

    def execute(self):
        stage = Path(f"C:/isaacsim_projects/grocery_ground_scene_seed_{self.seed}.usd")
        self.remember("stage", stage)
        if not stage.exists():
            self.run(
                "scene", "create-grocery-scene.ps1", ["-Seed", self.seed, "-Output", stage], 90
            )
        image = self.capture(str(stage))
        detections = self.remember(
            "detections",
            payload(self.run("dino", "test-perception-dino.ps1", ["-Image", image]), "result"),
        )
        geometries = {}
        for object_id in ("alphabet_soup", "basket"):
            segmentation = self.remember(
                object_id + "_segmentation",
                payload(
                    self.run(
                        "sam3-" + object_id,
                        "test-perception-sam3.ps1",
                        ["-Detections", detections, "-ObjectId", object_id],
                    ),
                    "result",
                ),
            )
            geometries[object_id] = self.remember(
                object_id + "_geometry",
                payload(
                    self.run(
                        "geometry-" + object_id,
                        "test-perception-geometry.ps1",
                        ["-Segmentation", segmentation],
                        30,
                    ),
                    "result",
                ),
            )
        self.remember(
            "pair",
            payload(
                self.run(
                    "pair",
                    "test-geometry-pair.ps1",
                    ["-Target", geometries["alphabet_soup"], "-Basket", geometries["basket"]],
                    30,
                ),
                "result",
            ),
        )
        self.execute_motion(stage, geometries)

    def execute_motion(self, stage, geometries):
        """Run gates/motion using saved perception; retained for bounded retries."""
        geometry_arguments = [
            "-Stage",
            stage,
            "-TargetGeometry",
            geometries["alphabet_soup"],
            "-BasketGeometry",
            geometries["basket"],
        ]
        output = self.run("feasibility", "test-perceived-feasibility.ps1", geometry_arguments, 110)
        diagnostic = re.findall(r"Saved diagnostic: (.+\.json)", output)[-1].strip()
        self.remember("feasibility", diagnostic)
        for mode in ("Grasp", "Graph"):
            arguments = [
                *geometry_arguments,
                "-Mode",
                mode,
                "-Seed",
                self.seed,
                "-PhysicsHz",
                120,
                "-Headless",
            ]
            if mode == "Graph":
                arguments += [
                    "-GraspResult",
                    self.state["artifacts"]["grasp"],
                    "-Graph",
                    self.graph,
                ]
            output = self.run(
                mode.lower(), "test-graph-pick-place.ps1", arguments, allow_failed_result=True
            )
            result_path = re.findall(r"Saved graph result: (.+\.json)", output)[-1].strip()
            self.remember(mode.lower(), result_path)
            result = json.loads(Path(result_path).read_text(encoding="utf-8"))
            if self.state["steps"][-1]["exit_code"] != 0:
                raise RuntimeError(f"{mode} process failed; result preserved: {result_path}")
            if not result.get("ok"):
                raise RuntimeError(f"{mode} failed; result: {result_path}")
            if mode == "Graph" and (
                result["graph_sha256"] != self.state["graph_sha256"]
                or result["evaluation"]["seed"] != self.seed
                or result["evaluation"]["status"] != "succeeded"
                or not all(result["evaluation"]["checkpoints"].values())
                or result["evaluation"]["disallowed_collisions"]
            ):
                raise RuntimeError("Graph/evaluator evidence failed independent receipt checks")
        self.state["status"] = "succeeded"
        self.save()
        print(f"seed {self.seed}: SUCCEEDED; {self.state['artifacts']['graph']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--seeds", nargs="+", type=int, default=list(range(3, 10)))
    selection.add_argument(
        "--resume-trial",
        type=Path,
        help="Retry gates using a failed trial's saved geometry; preserve original",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=GRAPH,
        help="Validated accepted.json to freeze for this entire batch",
    )
    args = parser.parse_args()
    resume = None
    if args.resume_trial:
        resume = json.loads(args.resume_trial.read_text(encoding="utf-8"))
        if resume["status"] != "failed":
            parser.error("Only a failed trial can be resumed")
        args.seeds = [resume["seed"]]
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in range(10) for seed in args.seeds):
        parser.error("Seeds must be distinct integers in 0..9")
    graph = args.graph.resolve()
    receipt = verify_accepted(graph)
    if resume and resume["graph_sha256"] != receipt["graph_sha256"]:
        parser.error("Retry must use the original frozen graph")
    print(f"Frozen graph: {graph}; SHA-256: {receipt['graph_sha256']}", flush=True)
    directory = (
        ROOT / "results/layout-trials" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    directory.mkdir(parents=True)
    print(f"Batch evidence: {directory}", flush=True)
    failures = []
    for seed in args.seeds:
        trial = Trial(seed, directory, graph)
        try:
            if resume:
                trial.state["retry_of"] = str(args.resume_trial.resolve())
                trial.state["artifacts"] = dict(resume["artifacts"])
                for key in ("feasibility", "grasp", "graph"):
                    trial.state["artifacts"].pop(key, None)
                trial.save()
                trial.execute_motion(
                    Path(resume["artifacts"]["stage"]),
                    {
                        name: resume["artifacts"][name + "_geometry"]
                        for name in ("alphabet_soup", "basket")
                    },
                )
            else:
                trial.execute()
        except Exception as error:
            trial.state.update(status="failed", error=f"{type(error).__name__}: {error}")
            trial.save()
            failures.append(seed)
            print(f"seed {seed}: FAILED: {error}", flush=True)
    print(f"Batch complete; failed seeds: {failures}; evidence: {directory}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
