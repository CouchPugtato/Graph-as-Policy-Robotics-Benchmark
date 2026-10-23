"""Pinned published GaP environments using the shared bounded LIBERO controller."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import partial

import numpy as np
import yaml

from .catalog import ROOT

REVISION = "fd2bc0f63369ba39137df018bbca8f6b372ffa0b"
SOURCE = ROOT / "cache/upstream/Variational-Automation-Benchmark"
PAN_TEMPLATE = (
    SOURCE
    / "libero/libero/assets/stable_scanned_objects/chefmate_8_frypan"
    / "visual/chefmate_8_frypan_vis.msh"
)
CASES = {
    "fulfillment": (
        "libero_object_all_variance",
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket.yaml",
    ),
    "packing": ("libero_object_packing", "pack_all_objects_v00.yaml"),
    "popcorn": ("libero_popcorn_production", "kitchen_scene9_popcorn_production.yaml"),
}


def definition(case):
    suite, filename = CASES[case]
    path = SOURCE / "tasks" / suite / filename
    actual = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if actual != REVISION:
        raise ValueError("published benchmark source differs from the GaP pinned revision")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if len(value["inits"]) < 10 or not (SOURCE / "libero/libero/assets/scenes").is_dir():
        raise ValueError("published benchmark initial states or assets are missing")
    return path, value


def prepare():
    if not SOURCE.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                "--no-checkout",
                "https://github.com/ehehee/Variational-Automation-Benchmark.git",
                str(SOURCE),
            ],
            check=True,
            timeout=180,
        )
        subprocess.run(
            ["git", "-C", str(SOURCE), "fetch", "--depth", "1", "origin", REVISION],
            check=True,
            timeout=180,
        )
        subprocess.run(
            ["git", "-C", str(SOURCE), "checkout", "--detach", REVISION], check=True, timeout=180
        )
    actual = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if actual != REVISION:
        raise ValueError("existing benchmark checkout was not changed because its revision differs")
    subprocess.run(
        ["git", "-C", str(SOURCE), "sparse-checkout", "set", "tasks", "libero/libero"],
        check=True,
        timeout=300,
    )
    return {
        "ok": True,
        "revision": REVISION,
        "task_files": [str(definition(c)[0]) for c in CASES],
        "models_loaded": False,
        "simulator_started": False,
    }


def task_manifest(case):
    from gap_libero.scenario_catalog import task_catalog

    path, task = definition(case)
    value = task_catalog("packing" if case == "packing" else "fulfillment")
    value["task_id"] = f"gap-published-{case}-local-qwen"
    value["instruction"] = (
        (
            "Place the frying pan onto the flat stove cooking surface"
            if case == "popcorn"
            else task["language"]
        )
        + ". Use the published task and its unchanged success predicate. "
        "Perceive the target and destination with local Qwen and RGB-D before each transfer. "
        "Use at least three physical action subgraphs. "
        "The basket-named transport actions use the selected destination including the stove. "
        + (
            "Process all six requested items through select_next_item and check_all_packed. "
            "The benchmark retires delivered objects automatically; never teleport them yourself."
            if case == "packing"
            else "The popcorn goal is pan placement only; no knob or heating actions are deployed."
            if case == "popcorn"
            else "The requested object is the alphabet soup can."
        )
    )
    value["runner"]["command"] = [
        "{root}/.venv-libero/Scripts/python.exe",
        "-m",
        "gap_libero.cli",
        "published",
        "--scenario",
        case,
        "--seed",
        "{seed}",
        "--graph",
        "{graph}",
        "--output",
        "{output}",
        "--enable-motion",
    ]
    value["runner"]["timeout_s"] = 300
    value["checkpoint_evidence"] = [
        "evaluation.task_success",
        "evaluation.checkpoints.target_grasped",
        "evaluation.checkpoints.target_lifted",
        "evaluation.checkpoints.target_released",
        "evaluation.checkpoints.target_contained",
    ]
    value["required_true"] = value["checkpoint_evidence"]
    value["max_trace_steps"] = 160
    value["limitations"] = [
        f"Published {task['id']} with GaP pinned source {REVISION}",
        "Ten saved initial states zero through nine of one task not the entire task suite",
        "Unchanged published success predicate with RGB-D camera and bounded controller overrides",
        "Packing retires delivered items using the upstream predicate "
        "not physical persistent containment",
        "Popcorn measures published pan placement only not the complete cooking sequence",
        "At most sixteen local Qwen requests and six thousand simulator steps per episode",
    ]
    for skill in value["skills"]:
        skill["instructions"] = (
            "Use only declared actions and real predicate outputs. "
            "Perceive before motion. No simulator object pose fallback or arbitrary code. "
            "Do not mutate task goals or move objects except through bounded robot actions."
        )
    destination = ROOT / "cache/harness-tasks" / f"published-{case}.json"
    text = json.dumps(value, indent=2) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.read_text(encoding="utf-8") != text:
        destination.write_text(text, encoding="utf-8")
    return destination


def binding_signature(case):
    path, _ = definition(case)
    files = [
        path,
        ROOT / "src/gap_libero/benchmarks/vab.py",
        ROOT / "src/gap_libero/actions.py",
        ROOT / "src/gap_libero/backend.py",
        ROOT / "src/gap_libero/vision.py",
        ROOT / "src/gap_libero/local_vlm.py",
        ROOT / "src/gap_libero/cli.py",
        ROOT / "src/gap_libero/scenario_catalog.py",
        ROOT / "src/gap_libero/pan_grasp.py",
    ]
    if case == "popcorn":
        files.append(PAN_TEMPLATE)
    digest = hashlib.sha256(REVISION.encode())
    for source in files:
        digest.update(source.read_bytes())
    return digest.hexdigest()


def readiness(case):
    path = ROOT / "cache/vab-readiness" / f"{case}.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if report.get("ok") is not True or report.get("binding_signature") != binding_signature(case):
        raise RuntimeError(
            f"published {case} bindings have not passed their current local smoke test; "
            "run python -m gap_libero.benchmarks.vab smoke before generation"
        )
    return report


def smoke(case):
    signature = binding_signature(case)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results/benchmarks" / f"published-{case}" / f"smoke-{stamp}"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ROOT / ".venv-libero/Scripts/python.exe"),
        "-m",
        "gap_libero.cli",
        "published",
        "--scenario",
        case,
        "--enable-motion",
        "--output",
        str(output),
    ]
    print(
        f"Checking published {case} bindings with one local reference graph no hosted calls",
        flush=True,
    )
    started = time.monotonic()
    with output.with_suffix(".log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            while True:
                try:
                    code = process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    print(f"published {case} smoke running {elapsed:.0f}s of 300s", flush=True)
                    if elapsed >= 300:
                        raise RuntimeError(
                            "published smoke worker exceeded its time budget"
                        ) from None
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    events = receipt.get("graph", {}).get("trace", [])
    failures = [event for event in events if event["outcome"] in ("failed", "false")]
    unchanged = signature == binding_signature(case)
    report = {
        "ok": code == 0 and receipt.get("ok") is True and unchanged,
        "binding_signature": signature,
        "case": case,
        "receipt": str(receipt_path),
        "completed_items": receipt.get("completed_items", []),
        "detail": (
            receipt.get("detail") or (failures[-1].get("detail") if failures else None)
            if unchanged
            else "binding files changed while this check was running rerun it"
        ),
        "hosted_model_calls": 0,
        "scope": "one reference graph one initial state",
    }
    marker = ROOT / "cache/vab-readiness" / f"{case}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def create_backend(case, seed, motion):
    if not 0 <= seed < 10:
        raise ValueError("published variation indices must be zero through nine")
    path, spec = definition(case)
    if any(name == "libero" or name.startswith("libero.") for name in sys.modules):
        raise RuntimeError("published environments require a fresh isolated worker")
    config = ROOT / "cache/benchmarks/config/vab"
    config.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(
        {
            "benchmark_root": str(SOURCE / "libero/libero"),
            "assets": str(SOURCE / "libero/libero/assets"),
        }
    )
    config_path = config / "config.yaml"
    if config_path.exists() and config_path.read_text(encoding="utf-8") != content:
        raise ValueError("existing published environment configuration differs")
    if not config_path.exists():
        config_path.write_text(content, encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    sys.path.insert(0, str(SOURCE / "libero"))
    importlib.invalidate_caches()
    from libero.vab import load_task
    from libero.vab.env import VABEnv

    from gap_libero.backend import LiberoBackend

    class RawEnvironment(VABEnv):
        def _load_model(self):
            super()._load_model()
            if case == "popcorn":
                compiler = self.model.root.find("compiler")
                if compiler is None:
                    import xml.etree.ElementTree as ET

                    compiler = ET.SubElement(self.model.root, "compiler")
                compiler.set("boundmass", "0.000001")
                compiler.set("boundinertia", "0.000000001")

        def _filter_obs(self, raw):
            return raw

        @property
        def objects_dict(self):
            return {obj.name: obj for obj in self._mujoco_objects}

    task = load_task(path)
    task.camera_width, task.camera_height, task.camera_depth = 640, 480, True
    task.cameras = ["agentview"]
    # the upstream GaP bridge also overrides camera settings and rehearsal horizon
    step_limit = 6000 if case == "packing" else 4000
    task.horizon = step_limit + 50
    env = RawEnvironment(task, ignore_done=False, hard_reset=False, render_gpu_device_id=0)

    class Wrapper:
        def __init__(self):
            self.env, self.sim, self.robots = env, env.sim, env.robots

        def step(self, action):
            return env.step(action)

        def check_success(self):
            return bool(env._check_success())

        def close(self):
            env.close()

    try:
        np.random.seed(seed)
        observation = env.reset(init_index=seed)
        if case == "popcorn":
            from gap_libero.pan_grasp import configure_pan_gripper

            configure_pan_gripper(env)
        backend = LiberoBackend(
            Wrapper(),
            motion_enabled=motion,
            max_steps=step_limit,
            episode_step_limit=step_limit,
            workspace=[[-0.8, -0.8, 0.005], [0.8, 0.8, 1.6]],
        )
        backend.observation = observation
        backend.check_workspace(backend.eef_position())
        backend.published_spec = spec
        backend.published_path = path
        return backend
    except BaseException:
        env.close()
        raise


def published_candidates(bundle, *, floor, kitchen):
    from gap_libero.vision import native_candidates

    regions = native_candidates(bundle, floor_z=floor, broad=True, split_color=kitchen)
    if kitchen:
        fine = native_candidates(
            bundle, floor_z=floor, broad=True, split_color=True, color_threshold=0.06
        )
        for item in fine:
            if item["pixels"] < 4000:
                continue
            if not any(
                (item["mask"] & other["mask"]).sum() / (item["mask"] | other["mask"]).sum() > 0.9
                for other in regions
            ):
                regions.append(item)
    return [{**item, "id": f"C{index + 1}"} for index, item in enumerate(regions[:10])]


def pan_handle_geometry(source, floor):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    # separate the low bowl from a stove that touches its image silhouette
    low = source[source[:, 2] < floor + 0.03]
    if len(low) < 30:
        raise ValueError("pan bowl surface is not sufficiently visible")
    _, keep = np.unique(np.floor(low / 0.003).astype(int), axis=0, return_index=True)
    low = low[keep]
    pairs = cKDTree(low).query_pairs(0.008, output_type="ndarray")
    adjacency = coo_matrix((np.ones(len(pairs)), pairs.T), shape=(len(low), len(low)))
    _, labels = connected_components(adjacency, directed=False)
    bowl = low[labels == np.bincount(labels).argmax()]
    spans = np.ptp(bowl[:, :2], axis=0)
    if len(bowl) < 30 or not (0.12 < min(spans) <= max(spans) < 0.23):
        raise ValueError("pan bowl does not match its visible size prior")
    center = (bowl[:, :2].min(axis=0) + bowl[:, :2].max(axis=0)) / 2
    relative = source[:, :2] - center
    best = None
    for angle in np.linspace(-np.pi, np.pi, 361):
        direction = np.array([np.cos(angle), np.sin(angle)])
        along = relative @ direction
        across = np.abs(relative @ np.array([-direction[1], direction[0]]))
        band = (along > 0.13) & (along < 0.24)
        selected = band & (across < 0.014)
        counts, _ = np.histogram(along[selected], bins=np.linspace(0.13, 0.24, 12))
        score = sum(counts > 2) - 0.001 * sum(band & (across > 0.025) & (across < 0.06))
        if best is None or score > best[0]:
            best = score, direction, selected
    score, direction, selected = best
    if score < 8:
        raise ValueError("pan handle grasp patch is not sufficiently visible")
    # use continuous narrow support rather than points belonging to the stove plane
    axis = direction
    handle = center + 0.17 * axis
    patch = source[np.linalg.norm(source[:, :2] - handle, axis=1) < 0.015]
    if len(patch) < 12:
        raise ValueError("pan handle grasp patch is not sufficiently visible")
    return (
        center,
        [*handle, float(np.median(patch[:, 2]))],
        float(np.arctan2(axis[1], axis[0]) - np.pi / 2),
    )


def stove_surface_geometry(source, floor):
    elevated = source[source[:, 2] > floor + 0.035]
    if len(elevated) < 100:
        raise ValueError("stove surface is not sufficiently visible")
    rng = np.random.default_rng(0)
    best = np.zeros(len(elevated), dtype=bool)
    for _ in range(128):
        a, b, c = elevated[rng.choice(len(elevated), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9 or abs(normal[2]) / length < 0.9:
            continue
        selected = np.abs((elevated - a) @ (normal / length)) < 0.0015
        if selected.sum() > best.sum():
            best = selected
    surface = elevated[best]
    if len(surface) < 100:
        raise ValueError("stove surface plane was not verified")
    low, high = np.quantile(surface[:, :2], [0.005, 0.995], axis=0)
    if not (0.13 < min(high - low) <= max(high - low) < 0.25):
        raise ValueError("stove surface does not match its visible size prior")
    return (low + high) / 2


def geometry(
    points,
    object_id,
    camera_position=None,
    *,
    silhouette_points=None,
    floor=0,
    objects=None,
    pan_template=None,
):
    cloud = np.asarray(points, dtype=float)
    if len(cloud) < 30 or not np.isfinite(cloud).all() or np.max(np.ptp(cloud, axis=0)) > 0.6:
        raise ValueError("published object needs a bounded visible RGB-D surface")
    source = cloud if silhouette_points is None else np.asarray(silhouette_points)
    top = float(np.quantile(cloud[:, 2], 0.98))
    if object_id == "basket":
        center = (source[:, :2].min(axis=0) + source[:, :2].max(axis=0)) / 2
        position, yaw = [*center, floor], 0.0
    elif object_id == "flat_stove":
        center = stove_surface_geometry(source, floor)
        center[0] -= 0.03
        position, yaw = [*center, top], 0.0
    elif object_id == "chefmate_8_frypan":
        spans = np.ptp(source[:, :2], axis=0)
        if min(spans) > 0.22 or max(spans) > 0.4:
            center, position, yaw = pan_handle_geometry(source, floor)
        else:
            xy = source[:, :2]
            center = np.median(xy, axis=0)
            far = xy[np.argmax(np.linalg.norm(xy - center, axis=1))]
            direction = (far - center) / np.linalg.norm(far - center)
            handle = far - 0.09 * direction
            patch = source[np.linalg.norm(xy - handle, axis=1) < 0.025]
            if len(patch) < 12:
                raise ValueError("pan handle grasp patch is not sufficiently visible")
            position = [*handle, float(np.median(patch[:, 2]))]
            yaw = float(np.arctan2(direction[1], direction[0]) - np.pi / 2)
    else:
        cap = source[source[:, 2] > top - 0.004]
        if len(cap) < 12:
            raise ValueError("target top surface is not sufficiently visible")
        center = (cap[:, :2].min(axis=0) + cap[:, :2].max(axis=0)) / 2
        obj = objects[object_id]
        height = float(obj.top_offset[2] - obj.bottom_offset[2])
        if not 0.02 <= height <= 0.3:
            raise ValueError("unsupported published object dimension prior")
        position = [*center, top - height / 2]
        covariance = np.cov(cap[:, :2].T)
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, -1]
        yaw = float(np.arctan2(direction[1], direction[0])) if values[-1] > 1.4 * values[0] else 0.0
    result = {
        "position_m": list(map(float, position)),
        "yaw_rad": yaw,
        "point_count": len(cloud),
        "uses_evaluator_pose": False,
        "geometry_source": "local-qwen-selected-published-rgbd",
        "top_surface_m": top,
    }
    if object_id == "chefmate_8_frypan":
        result["object_center_xy_m"] = center.tolist()
        if pan_template is not None:
            from gap_libero.pan_grasp import fit_pan_pose, load_pan_template

            initial_center, _, initial_yaw = pan_handle_geometry(source, floor)
            fit = fit_pan_pose(
                source,
                load_pan_template(pan_template),
                initial_center,
                initial_yaw + np.pi / 2,
                floor,
            )
            result["pan_fit"] = fit
            result["object_center_xy_m"] = fit["body_position_m"][:2]
    return result


def jaw_yaw(object_id, observed_yaw, current_yaw, *, thin=False, kitchen=False):
    from gap_libero.grocery_comparison import closest_parallel_jaw_yaw

    if kitchen:
        return closest_parallel_jaw_yaw(observed_yaw, np.pi / 2)
    if object_id in ("alphabet_soup", "tomato_sauce", "salad_dressing"):
        return 0.0
    reference = np.pi / 2 if thin else current_yaw
    return closest_parallel_jaw_yaw(observed_yaw + np.pi / 2, reference)


def run(output, graph, case, seed=0, motion=False, record=False):
    from gap_isaac.graph_policy import (
        GraphExecutionBoundExceeded,
        GraphPolicy,
        GraphPolicyExecutor,
        NodeResult,
    )
    from gap_libero.actions import LiberoGraphActions, RunPlan
    from gap_libero.local_vlm import LocalVLM, renderer_headroom
    from gap_libero.scenario_runner import SparseRecorder
    from gap_libero.vision import VisionActions

    graph_bytes = graph.read_bytes()
    source_signature = binding_signature(case)
    policy = GraphPolicy.from_json(graph)
    settings = LocalVLM().preflight() if motion else None
    if motion:
        renderer_headroom()
    backend = create_backend(case, seed, motion)
    spec = backend.published_spec
    requested = (
        spec["success"]["args"]["objs"] if case == "packing" else [spec["success"]["args"]["obj"]]
    )
    if case == "packing":

        def category(name):
            obj = backend.target_object(name)
            height = float(obj.top_offset[2] - obj.bottom_offset[2])
            return 0 if height > 0.12 else 1 if height > 0.055 else 2

        requested = sorted(requested, key=category)
    goal = "flat_stove" if case == "popcorn" else "basket"
    floor = 0.9 if case == "popcorn" else 0.0
    verification_positions = {}

    def capture_for_perception():
        bundle = backend.capture()
        # keep evaluator evidence separate from the rgbd returned to perception
        verification_positions[bundle.observation.rgb_sha256] = {
            name: backend.env.sim.data.get_body_xpos(backend.target_object(name).root_body).copy()
            for name in requested
        }
        return bundle

    class Actions:
        def __init__(self):
            self.index, self.target, self.cycle, self.calls = 0, requested[0], 0, 0
            self.vision = self.bindings = None
            self.checked = {
                "target_grasped": False,
                "target_lifted": False,
                "target_released": False,
                "target_contained": False,
            }
            self.completed = []
            self.guard_target()

        def guard_target(self):
            env, model = backend.env.env, backend.env.sim.model
            obstacles = {name for obj in env.objects_dict.values() for name in obj.contact_geoms}
            obstacles.update(
                name
                for name in model.geom_names
                if name and ("floor" in name or "table_collision" in name)
            )
            arm = [
                name
                for name in model.geom_names
                if name
                and (
                    name.startswith("robot0_")
                    and "link0" not in name
                    and "link1" not in name
                    or name.startswith("gripper0_")
                    and "finger" not in name
                )
            ]
            fingers = [
                name
                for name in model.geom_names
                if name and name.startswith("gripper0_") and "finger" in name
            ]
            non_target = obstacles - set(env.objects_dict[self.target].contact_geoms)
            pairs = [(a, b) for a in arm for b in obstacles]
            pairs += [(a, b) for a in fingers for b in non_target]
            backend.forbidden_pairs = {
                frozenset((model.geom_name2id(a), model.geom_name2id(b))) for a, b in pairs
            }
            backend.check_contacts()

        def delivered(self):
            return set(getattr(backend.env.sim, "_packing_state", {}).get("delivered", ()))

        def select(self, state):
            if self.index == len(requested):
                return NodeResult("false", "all published items visited")
            self.target = requested[self.index]
            self.index += 1
            self.bindings = None
            self.guard_target()
            return NodeResult("true", outputs={"target_id": self.target})

        def observe(self, state):
            if self.vision:
                self.calls += self.vision.identifier.calls
            if self.calls + 2 > 16:
                raise RuntimeError("published local vision call budget exhausted")
            self.vision = VisionActions(
                capture_for_perception,
                output / f"perception-{self.cycle:02d}",
                identifier=LocalVLM(max_candidates=10),
                candidate_provider=partial(
                    published_candidates, floor=floor, kitchen=case == "popcorn"
                ),
                descriptions={
                    **{
                        name: "the " + name.replace("_", " ")
                        for name in backend.env.env.objects_dict
                    },
                    "alphabet_soup": (
                        "the BLUE cylindrical can with a YELLOW lower band and dark lid. "
                        "Its visible side is BLUE not GREEN or RED. "
                        "Do not select the red and green can. "
                        "Judge visible colors not product names"
                    ),
                    "basket": "the open woven wicker basket",
                    "cream_cheese": "the small silver and blue Philadelphia cream cheese package",
                    "butter": (
                        "the rectangular butter package with a red and yellow label and "
                        "white blue top not the silver blue cream cheese package"
                    ),
                    "chefmate_8_frypan": "the frying pan with a long black handle",
                    "flat_stove": (
                        "the rectangular silver metal electric hotplate with a dark round burner "
                        "not the frying pan or its long handle"
                    ),
                },
                geometry_estimator=partial(
                    geometry,
                    floor=floor,
                    objects=backend.env.env.objects_dict,
                    pan_template=PAN_TEMPLATE if case == "popcorn" else None,
                ),
                silhouette_objects=set(backend.env.env.objects_dict),
            )
            self.cycle += 1
            self.bindings = None
            return self.vision.observe(state)

        def perceive(self, verb, container, state):
            if self.vision is None:
                raise RuntimeError("observe before perception")
            name = goal if container else self.target
            fn = {
                "detect": self.vision.detect,
                "identify": self.vision.identify,
                "segment": self.vision.segment,
                "estimate": self.vision.geometry,
                "check": self.vision.found,
            }[verb]
            return fn(name, state)

        def plan(self):
            if self.vision is None or not all(
                self.vision.geometries.get(n, {}).get("valid") for n in (self.target, goal)
            ):
                raise RuntimeError("published motion requires identified RGB-D geometry")
            target, destination = [
                self.vision.geometries[n]["position_m"] for n in (self.target, goal)
            ]
            pan_plan = None
            if case == "popcorn":
                from gap_libero.pan_grasp import PanPosePlanner

                fit = self.vision.geometries[self.target]["pan_fit"]
                stove_center = np.asarray(destination[:2]) + [0.03, 0]
                pan_plan = PanPosePlanner(backend, floor, stove_center, destination[2]).choose(fit)
                print(
                    f"Pan grasp planner verified {pan_plan['candidates_verified']} candidates",
                    flush=True,
                )
                target = pan_plan["position_m"]
                self.vision.geometries[self.target]["position_m"] = target
                (output / "pan-grasp-plan.json").write_text(json.dumps(pan_plan, indent=2))
                offset = (
                    np.asarray(target[:2])
                    - self.vision.geometries[self.target]["object_center_xy_m"]
                )
                destination = [*(np.asarray(destination[:2]) + offset).tolist(), destination[2]]
            clearance = floor + (0.2 if case == "popcorn" else 0.25)
            grasp = (
                float(target[2])
                if case == "popcorn"
                else max(
                    floor + 0.012, self.vision.geometries[self.target]["top_surface_m"] - 0.015
                )
            )
            if self.target == "milk":
                grasp = self.vision.geometries[self.target]["top_surface_m"] - 0.025
            if self.vision.geometries[self.target]["top_surface_m"] < floor + 0.04:
                grasp = max(grasp, floor + 0.018)
            orientation = backend.eef_orientation()
            yaw = jaw_yaw(
                self.target,
                self.vision.geometries[self.target]["yaw_rad"],
                2 * np.arctan2(orientation[1], orientation[0]),
                thin=self.vision.geometries[self.target]["top_surface_m"] < floor + 0.04,
                kitchen=case == "popcorn",
            )
            quaternion = [float(np.cos(yaw / 2)), float(np.sin(yaw / 2)), 0, 0]
            if pan_plan:
                quaternion = pan_plan["orientation_xyzw"]

            def move(position, holding=False):
                return {
                    "kind": "move",
                    "position_m": list(map(float, position)),
                    "orientation_xyzw": quaternion,
                    "max_steps": 400,
                    "position_gain": 3,
                    "maximum_translation_command": 0.15,
                    "tolerance_m": 0.004,
                    "require_grasp": holding,
                }

            actions = {
                "open_gripper": {"kind": "open", "max_steps": 25},
                "align_over_target": move(
                    pan_plan["hover_m"] if pan_plan else [*target[:2], clearance]
                ),
                "descend_to_grasp": move([*target[:2], grasp]),
                "close_gripper": {"kind": "close", "max_steps": 80, "preload_steps": 8},
                "lift_grasped": move([*target[:2], clearance], True),
                "move_over_basket": move([*destination[:2], clearance], True),
                "position_for_release": move(
                    [*destination[:2], destination[2] + 0.08 if case == "popcorn" else 0.24], True
                ),
                "release_object": {"kind": "open", "max_steps": 45},
                "retract_from_basket": move([*destination[:2], clearance]),
                "settle_release": {"kind": "hold", "max_steps": 30},
                "return_ready": move([*destination[:2], clearance]),
            }
            if pan_plan:
                actions["close_gripper"]["max_steps"] = 160
                body_xy = np.asarray(fit["body_position_m"][:2])
                separation = body_xy - stove_center
                overlap = np.linalg.norm(np.maximum(np.abs(separation) - 0.095, 0)) < 0.12
                pull = 0.06 * separation / np.linalg.norm(separation)
                pull[0] = np.clip(pull[0], -0.02, 0.02)
                withdrawal = np.asarray(target[:2]) + pull
                actions["unhook_pan"] = move([*withdrawal, grasp + 0.015], True)
                actions["lift_grasped"] = move(
                    [*(withdrawal if overlap else np.asarray(target[:2]) - [0.04, 0]), clearance],
                    True,
                )
                from scipy.spatial.transform import Rotation

                turn = Rotation.from_euler("z", np.pi / 2)
                transported = turn * Rotation.from_quat(quaternion)
                rotated_offset = turn.apply([*offset, 0])[:2]
                transport_xy = stove_center - [0.03, 0] + rotated_offset
                for name in (
                    "move_over_basket",
                    "position_for_release",
                    "retract_from_basket",
                    "return_ready",
                ):
                    actions[name]["position_m"][:2] = transport_xy.tolist()
                    actions[name]["orientation_xyzw"] = transported.as_quat().tolist()
                self.pan_needs_withdrawal = overlap
            plan = RunPlan(
                suite=CASES[case][0],
                task_id=0,
                seed=seed,
                target_object=self.target,
                workspace_m=backend.workspace.tolist(),
                max_episode_steps=1000,
                grasp_reference_offset_m=pan_plan["model_grasp_offset_m"]
                if pan_plan
                else [0, 0, 0],
                maximum_grasp_rotation_rad=0.25 if case == "popcorn" else None,
                actions=actions,
            )
            (output / f"plan-{self.cycle:02d}.json").write_text(plan.model_dump_json(indent=2))
            return plan

        def motion(self, name, state):
            if self.bindings is None:
                owner = self

                class Bindings(LiberoGraphActions):
                    def execute_action(self, name, state):
                        if (
                            case == "popcorn"
                            and name == "lift_grasped"
                            and owner.pan_needs_withdrawal
                        ):
                            unhook = super().execute_action("unhook_pan", state)
                            if unhook.outcome != "succeeded":
                                return unhook
                        if case == "popcorn" and name == "descend_to_grasp":
                            from gap_libero.pan_grasp import preshape_pan_gripper

                            preshape_pan_gripper(self)
                        if case == "popcorn" and name == "close_gripper":
                            from gap_libero.pan_grasp import settle_pan_grasp

                            return settle_pan_grasp(self)
                        return super().execute_action(name, state)

                    def check_hold(self):
                        if case == "packing" and self.plan.target_object in owner.delivered():
                            return
                        super().check_hold()

                self.bindings = Bindings(backend, self.plan())
            if (
                case == "packing"
                and self.target in self.delivered()
                and name in ("move_over_basket", "position_for_release")
            ):
                return NodeResult(detail="upstream predicate has retired the delivered item")
            result = self.bindings.timed_action(name, state)
            if result.outcome == "succeeded":
                if name == "close_gripper":
                    self.checked["target_grasped"] = backend.grasped(self.target)
                if name == "lift_grasped":
                    self.checked["target_lifted"] = backend.grasped(self.target)
            return result

        def placed(self, state):
            success = (
                self.target in self.delivered() if case == "packing" else backend.task_success()
            )
            released = not backend.grasped(self.target)
            self.checked.update(target_contained=success, target_released=released)
            if success and released and self.target not in self.completed:
                self.completed.append(self.target)
            return NodeResult(
                "true" if success and released else "false",
                outputs={"placement_verified": bool(success and released)},
            )

        def registry(self):
            from gap_libero.grocery_comparison import MOTION_ACTIONS

            registry = {
                name: partial(self.motion, name) for name in (*MOTION_ACTIONS, "return_ready")
            }
            registry.update(
                observe_scene=self.observe,
                select_next_item=self.select,
                check_grasp=lambda state: NodeResult(
                    "true" if backend.grasped(self.target) else "false"
                ),
                check_placement=self.placed,
                check_all_packed=lambda state: NodeResult(
                    "true"
                    if backend.task_success() and set(self.completed) == set(requested)
                    else "false"
                ),
            )
            for subject in ("target", "container"):
                for verb in ("detect", "identify", "segment", "estimate", "check"):
                    registry[f"{verb}_{subject}"] = partial(
                        self.perceive, verb, subject == "container"
                    )
            return registry

    try:
        if record:
            backend.recorder = SparseRecorder(backend, output / "recording")
            backend.recorder.record(backend.observation, 0)
        if motion:
            for _ in range(100 if case == "popcorn" else 20):
                backend.step([0, 0, 0, 0, 0, 0, -1])
        backend.capture().save(output / "initial")
        metadata = {
            "backend": "libero",
            "environment_family": "published-gap",
            "scenario": case,
            "suite": CASES[case][0],
            "task_id": 0,
            "seed": seed,
            "init_state_id": seed,
            "custom_task": False,
            "source_revision": REVISION,
            "binding_signature": source_signature,
            "task_definition_sha256": hashlib.sha256(
                backend.published_path.read_bytes()
            ).hexdigest(),
            "published_success": spec["success"],
            "published_horizon": spec["horizon"],
            "execution_horizon": backend.max_steps,
            "camera_override": "agentview RGB-D 640x480",
            "gripper_drive_gain_multiplier": 2 if case == "popcorn" else 1,
            "gripper_pad_contact_dimension": 4 if case == "popcorn" else 3,
            "scope": "one published task ten saved states not a whole suite score",
            "local_vlm": settings,
            "motion_enabled": motion,
            "models_loaded_by_worker": False,
            "mujoco_compiler_bounds": (
                {"boundmass": 1e-6, "boundinertia": 1e-9} if case == "popcorn" else None
            ),
        }
        if not motion:
            return {"ok": True, **metadata, "simulation_evaluated": False}
        if backend.task_success():
            raise RuntimeError("published goal already satisfied at reset")
        actions = Actions()
        trajectory = []

        def wrap(name, function):
            def call(state):
                start = backend.steps
                print(
                    f"published {case} | {actions.target} | {name} | "
                    f"step {start}/{backend.max_steps}",
                    flush=True,
                )
                try:
                    result = function(state)
                except Exception as error:
                    result = NodeResult("failed", f"{type(error).__name__}: {error}")
                if result.outcome == "failed" or (
                    result.outcome == "false" and name != "select_next_item"
                ):
                    domain = (
                        "geometry"
                        if name.startswith(("estimate_", "check_target", "check_container"))
                        else "perception"
                        if name.startswith(("observe_", "detect_", "identify_", "segment_"))
                        else "placement"
                        if name in ("check_placement", "check_all_packed")
                        else "control"
                    )
                    result = NodeResult(
                        result.outcome,
                        result.detail,
                        outputs={**(result.outputs or {}), "failure_domain": domain},
                    )
                trajectory.append(
                    {
                        "action": name,
                        "start_step": start,
                        "step": backend.steps,
                        "outcome": result.outcome,
                    }
                )
                return result

            return call

        bound_exceeded = False
        try:
            result = GraphPolicyExecutor(
                policy, {n: wrap(n, f) for n, f in actions.registry().items()}, max_steps=160
            ).execute({"seed": seed})
        except GraphExecutionBoundExceeded as error:
            result = error.result
            bound_exceeded = True
        graph_result = result.as_dict()
        localization_diagnostic = None
        if case != "popcorn" and actions.vision and result.status != "succeeded":
            estimate = actions.vision.geometries.get(actions.target, {})
            observed = verification_positions.get(estimate.get("rgb_sha256"), {}).get(
                actions.target
            )
            if estimate.get("valid") and observed is not None:
                error_xy = float(
                    np.linalg.norm(np.asarray(estimate["position_m"][:2]) - observed[:2])
                )
                localization_diagnostic = {
                    "target_xy_error_at_capture_m": error_xy,
                    "used_for_motion": False,
                }
                if error_xy > 0.06 and graph_result["trace"]:
                    last = graph_result["trace"][-1]
                    last["outputs"] = dict(last.get("outputs") or {})
                    last["outputs"]["failure_domain"] = "perception"
                    last["outputs"]["post_run_localization_diagnostic"] = localization_diagnostic
        if bound_exceeded:
            graph_result["execution_bound_exceeded"] = True
        if graph.read_bytes() != graph_bytes:
            raise RuntimeError("the graph changed during execution")
        if binding_signature(case) != source_signature:
            raise RuntimeError("published bindings changed during execution")
        success = backend.task_success()
        backend.capture().save(output / "final")
        model, data = backend.env.sim.model, backend.env.sim.data
        debug_geoms = [
            *backend.target_object(actions.target).contact_geoms,
            *backend.env.robots[0].gripper.contact_geoms,
        ]
        final_diagnostics = {
            "eef_position_m": backend.eef_position().tolist(),
            "target_body_position_m": data.get_body_xpos(
                backend.target_object(actions.target).root_body
            ).tolist(),
            "geometries": {
                name: {
                    "position_m": data.geom_xpos[model.geom_name2id(name)].tolist(),
                    "half_size_m": model.geom_size[model.geom_name2id(name)].tolist(),
                }
                for name in debug_geoms
            },
            "contacts": [
                [model.geom_id2name(c.geom1), model.geom_id2name(c.geom2)]
                for c in data.contact[: min(data.ncon, 64)]
            ],
        }
        return {
            **metadata,
            "ok": result.status == "succeeded" and success and all(actions.checked.values()),
            "simulation_evaluated": True,
            "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
            "graph": graph_result,
            "steps": backend.steps,
            "trajectory": trajectory,
            "perception": {
                "uses_oracle_object_poses": False,
                "local_vlm_calls": actions.calls
                + (actions.vision.identifier.calls if actions.vision else 0),
            },
            "evaluation": {
                "task_success": success,
                "localization_diagnostic": localization_diagnostic,
                "checkpoints": actions.checked,
                "disallowed_collisions": backend.disallowed_collisions,
            },
            "completed_items": actions.completed,
            "final_diagnostics": final_diagnostics,
        }
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check", "smoke"))
    parser.add_argument("--scenario", choices=CASES)
    args = parser.parse_args()
    if args.mode == "smoke":
        reports = [smoke(case) for case in ([args.scenario] if args.scenario else CASES)]
        sys.exit(0 if all(row["ok"] for row in reports) else 1)
    print(
        json.dumps(
            prepare()
            if args.mode == "prepare"
            else {"ok": True, "tasks": [str(task_manifest(c)) for c in CASES]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
