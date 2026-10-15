"""Optional LIBERO diagnostics capture and graph runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import traceback
from importlib.machinery import PathFinder
from pathlib import Path

import yaml


def configure(source: Path, output: Path) -> dict:
    root = source.resolve() / "libero" / "libero"
    paths = {"benchmark_root": root, "bddl_files": root / "bddl_files",
             "init_states": root / "init_files", "assets": root / "assets",
             "datasets": source.resolve() / "datasets"}
    for name in ("benchmark_root", "bddl_files", "init_states", "assets"):
        if not paths[name].is_dir():
            raise ValueError(f"LIBERO source is missing {name}")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "config.yaml"
    if destination.exists():
        raise FileExistsError("existing LIBERO configuration was not overwritten")
    destination.write_text(yaml.safe_dump({key: str(value) for key, value in paths.items()}),
                           encoding="utf-8")
    return {"ok": True, "config": str(destination.resolve()), "simulator_started": False}


def check() -> dict:
    packages = {}
    for name in ("libero", "robosuite", "mujoco", "bddl", "numpy", "torch", "future",
                 "matplotlib", "h5py"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    config = os.environ.get("LIBERO_CONFIG_PATH", "")
    # inspect the namespace without importing simulator code or triggering setup prompts
    parent = PathFinder.find_spec("libero", sys.path)
    child = (PathFinder.find_spec("libero.libero", parent.submodule_search_locations)
             if parent and parent.submodule_search_locations else None)
    import_path_ready = bool(child and child.origin and Path(child.origin).is_file())
    return {"ok": all(packages.values()) and import_path_ready and bool(config)
            and (Path(config) / "config.yaml").is_file(),
            "packages": packages, "config_directory_set": bool(config),
            "libero_import_path_ready": import_path_ready,
            "simulator_started": False, "runtime_validated": False,
            "models_loaded": False, "wsl_required": "not established"}


def execute(args: argparse.Namespace) -> dict:
    if args.mode == "compare-grocery":
        from gap_libero.grocery_comparison import DEFAULT_GRAPH, compare_grocery

        return compare_grocery(args.output, args.graph or DEFAULT_GRAPH, args.seed,
                               capture_only=not args.enable_motion)
    from gap_libero.backend import create_backend

    if args.mode == "run":
        from gap_isaac.graph_policy import GraphPolicy
        from gap_libero.actions import RunPlan, run_graph

        if not args.enable_motion:
            raise ValueError("run requires --enable-motion")
        plan = RunPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
        policy = GraphPolicy.from_json(args.graph)
        missing = {node.action for node in policy.nodes.values()} - set(plan.actions)
        if missing:
            raise ValueError(f"plan lacks graph actions: {sorted(missing)}")
        backend, metadata = create_backend(
            suite=plan.suite, task_id=plan.task_id, init_state_id=plan.init_state_id,
            seed=plan.seed, motion_enabled=True, workspace=plan.workspace_m,
            max_steps=plan.max_episode_steps, forbidden_geom_pairs=plan.forbidden_geom_pairs)
    else:
        backend, metadata = create_backend(suite=args.suite, task_id=args.task_id,
                                           init_state_id=args.init_state_id, seed=args.seed)
    try:
        if args.mode == "capture":
            bundle = backend.capture()
            bundle.save(args.output / "observation")
            result = {"ok": True, "bundle": str((args.output / "observation").resolve()),
                      "depth": bundle.depth_summary(), "motion_enabled": False,
                      "target_objects": sorted(backend.env.env.objects_dict)}
        else:
            result = run_graph(backend, policy, plan)
            result["graph_sha256"] = hashlib.sha256(args.graph.read_bytes()).hexdigest()
            result["plan_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
        return {**result, **metadata, "models_loaded": False}
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("check")
    commands.add_parser("prepare-runtime")
    config = commands.add_parser("configure")
    config.add_argument("--source", type=Path, required=True)
    config.add_argument("--output", type=Path, required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--suite", default="libero_spatial")
    capture.add_argument("--task-id", type=int, default=0)
    capture.add_argument("--init-state-id", type=int, default=0)
    capture.add_argument("--seed", type=int, default=0)
    run = commands.add_parser("run")
    run.add_argument("--graph", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--enable-motion", action="store_true")
    grocery = commands.add_parser("compare-grocery")
    grocery.add_argument("--seed", type=int, choices=range(10), default=0)
    grocery.add_argument("--graph", type=Path)
    grocery.add_argument("--enable-motion", action="store_true")
    for command in (capture, run, grocery):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "check":
        result = check()
    elif args.mode == "prepare-runtime":
        from gap_libero.windows_runtime import prepare_windows_runtime

        result = prepare_windows_runtime()
    elif args.mode == "configure":
        result = configure(args.source, args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        try:
            result = execute(args)
        except Exception as error:
            result = {"ok": False, "backend": "libero", "error_type": type(error).__name__,
                      "detail": str(error), "traceback": traceback.format_exc(),
                      "models_loaded": False}
        (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
