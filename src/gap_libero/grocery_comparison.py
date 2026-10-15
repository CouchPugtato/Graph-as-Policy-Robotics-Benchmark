"""Run an unchanged generated graph against the matched custom LIBERO grocery task."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from gap_isaac.graph_policy import GraphPolicy, GraphPolicyExecutor, NodeResult
from gap_libero.actions import LiberoGraphActions, RunPlan
from gap_libero.backend import LiberoBackend

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "graphs/generated/codex-20261009T145742Z-f28cf576/accepted.json"


def closest_parallel_jaw_yaw(yaw: float, reference: float = 0.0) -> float:
    if not math.isfinite(yaw) or not math.isfinite(reference):
        raise ValueError("jaw orientation angles must be finite")
    # swapping the two identical jaws leaves the basket alignment unchanged
    difference = (yaw - reference + math.pi / 2) % math.pi - math.pi / 2
    return reference + difference


def create_grocery_backend(seed=0, motion_enabled=False, max_steps=1000, workspace=None,
                           **settings):
    from libero.libero.envs import OffScreenRenderEnv

    from gap_libero import grocery_task  # noqa: F401 - registers the custom libero problem

    env = OffScreenRenderEnv(bddl_file_name=str(ROOT / "libero/grocery.bddl"),
        scene_seed=seed, robots=["OnTheGroundPanda"], initialization_noise=None,
        camera_names=["agentview"], camera_widths=320, camera_heights=240, camera_depths=True,
        controller="OSC_POSE", control_freq=20, horizon=max_steps + 50,
        hard_reset=False, ignore_done=False, render_gpu_device_id=0)
    try:
        env.seed(seed)
        env.reset()
        for _ in range(40):
            observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        pairs = env.env.forbidden_pairs()
        pairs.extend(settings.pop("forbidden_geom_pairs", ()))
        backend = LiberoBackend(env, motion_enabled=motion_enabled, max_steps=max_steps,
            workspace=workspace or [[-0.35, -0.85, 0.02], [1.0, 0.85, 1.1]],
            forbidden_geom_pairs=pairs, **settings)
        backend.observation = observation
        backend.check_workspace(backend.eef_position())
        backend.check_contacts()
    except BaseException:
        env.close()
        raise
    return backend, {"backend": "libero", "task": "grocery_ground", "seed": seed,
        "custom_task": True, "official_libero_benchmark": False,
        "scene_layout": "ground-v1", "geometry_source": "oracle-scene-known",
        "controller": "OSC_POSE", "settle_time_s": 2.0,
        "scene": asdict(env.env.scene_plan), "models_loaded": False}


def build_action_plan(backend, seed):
    env = backend.env.env
    data = env.sim.data
    target = data.get_body_xpos(env.objects_dict["alphabet_soup"].root_body).copy()
    basket = data.get_body_xpos(env.objects_dict["basket"].root_body).copy()
    grasp_z = float(target[2] + 0.035)
    transit_z = 0.28
    orientation = [1, 0, 0, 0]
    basket_rotation = data.get_body_xmat(env.objects_dict["basket"].root_body)
    basket_yaw = math.atan2(basket_rotation[1, 0], basket_rotation[0, 0])
    approach_heading = math.atan2(float(basket[1]), float(basket[0]))
    basket_yaw = closest_parallel_jaw_yaw(basket_yaw, approach_heading)
    basket_orientation = [math.cos(basket_yaw / 2), math.sin(basket_yaw / 2), 0, 0]

    def move(position, holding=False):
        return {"kind": "move", "position_m": position, "orientation_xyzw": orientation,
                "max_steps": 400, "tolerance_m": 0.004, "position_gain": 1,
                "maximum_translation_command": 0.15,
                "require_grasp": holding}

    plan = {"schema_version": 1, "suite": "grocery_ground",
        "task_id": 0, "seed": seed, "target_object": "alphabet_soup",
        "workspace_m": backend.workspace.tolist(), "max_episode_steps": backend.max_steps,
        "actions": {
            "open_gripper": {"kind": "open", "max_steps": 25},
            "align_over_target": move([float(target[0]), float(target[1]), transit_z]),
            "descend_to_grasp": move([float(target[0]), float(target[1]), grasp_z]),
            "close_gripper": {"kind": "close", "max_steps": 80, "preload_steps": 8},
            "lift_grasped": move([float(target[0]), float(target[1]), transit_z], True),
            "move_over_basket": move([float(basket[0]), float(basket[1]), transit_z], True),
            "position_for_release": move([float(basket[0]), float(basket[1]), 0.115], True),
            "release_object": {"kind": "open", "max_steps": 45},
            "retract_from_basket": move([float(basket[0]), float(basket[1]), transit_z]),
            "settle_release": {"kind": "verify", "max_steps": 40},
        }}
    for name in ("move_over_basket", "position_for_release", "retract_from_basket"):
        plan["actions"][name]["orientation_xyzw"] = basket_orientation
    plan["actions"]["position_for_release"]["tolerance_m"] = 0.0015
    plan["actions"]["position_for_release"]["position_gain"] = 4
    return RunPlan.model_validate(plan)


def compare_grocery(output: Path, graph: Path = DEFAULT_GRAPH, seed=0, capture_only=False,
                    record=False):
    graph_bytes = graph.read_bytes()
    policy = GraphPolicy.from_json(graph)
    backend, metadata = create_grocery_backend(seed, motion_enabled=not capture_only)
    try:
        if record:
            backend.start_recording(output / "recording")
            metadata["recording"] = {"frames": "recording/frames.json", "fps": 20}
        sources = ["src/gap_libero/grocery_task.py", "src/gap_libero/grocery_comparison.py",
                   "src/gap_libero/actions.py", "src/gap_libero/backend.py",
                   "src/gap_isaac/grocery_scene.py", "src/gap_isaac/sampling.py",
                   "src/gap_isaac/evaluate.py", "configs/tasks/grocery_fulfillment.yaml",
                   "libero/grocery.bddl"]
        metadata["implementation_files_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
        metadata["containment_evaluator"] = "gap_isaac.evaluate.target_is_contained"
        metadata["contact_parameters"] = {"proxy_margin_m": 0, "proxy_solref": [0.01, 1.0],
                                         "finger_force_limit_n": 7.2}
        (output / "scene.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        backend.capture().save(output / "initial")
        if capture_only:
            return {"ok": True, **metadata, "motion_enabled": False,
                    "initial_observation": str(output / "initial")}
        plan = build_action_plan(backend, seed)
        plan_text = plan.model_dump_json(indent=2)
        (output / "plan.json").write_text(plan_text, encoding="utf-8")
        bindings = LiberoGraphActions(backend, plan)
        registry = bindings.registry()
        checkpoints = {"target_grasped": False, "target_lifted": False,
                       "target_released": False, "target_contained": False}
        trajectory = []
        initial_height = float(backend.env.env.containment()["target_position_m"][2])

        def phase_state():
            data, model = backend.env.sim.data, backend.env.sim.model
            return {"step": backend.steps,
                "eef_position_m": backend.eef_position().tolist(),
                "grasped": backend.grasped("alphabet_soup"),
                "objects": {name: {
                    "position_m": data.get_body_xpos(obj.root_body).tolist(),
                    "rotation_matrix": data.get_body_xmat(obj.root_body).tolist(),
                    "linear_velocity_m_s": data.get_body_xvelp(obj.root_body).tolist()}
                    for name, obj in backend.env.env.objects_dict.items()},
                "contacts": [{"geometries": [model.geom_id2name(contact.geom1),
                                              model.geom_id2name(contact.geom2)],
                              "distance_m": float(contact.dist)}
                             for contact in data.contact[:data.ncon]],
                **backend.env.env.containment()}

        initial_state = phase_state()

        def wrapped(name, action):
            def execute(state):
                print(f"LIBERO graph action: {name}", flush=True)
                before = phase_state()
                if name == "settle_release":
                    for _ in range(40):
                        bindings.command()
                try:
                    result = action(state)
                except Exception as error:
                    result = NodeResult("failed", f"{type(error).__name__}: {error}")
                values = backend.env.env.containment()
                held = backend.grasped("alphabet_soup")
                if name == "close_gripper" and result.outcome == "succeeded":
                    checkpoints["target_grasped"] = held
                if name == "lift_grasped" and result.outcome == "succeeded":
                    checkpoints["target_lifted"] = (
                        held and values["target_position_m"][2] >= initial_height + 0.03)
                    if not checkpoints["target_lifted"]:
                        result = NodeResult("failed", "target did not lift with a verified grasp")
                if name in ("release_object", "settle_release"):
                    checkpoints["target_released"] = bool(values["target_released"])
                    checkpoints["target_contained"] = bool(values["target_contained"])
                trajectory.append({"action": name, "outcome": result.outcome,
                    "start_step": before["step"], "step": backend.steps,
                    "eef_position_m": backend.eef_position().tolist(),
                    "grasped": held, "before": before, "after": phase_state(), **values})
                return result
            return execute

        registry = {name: wrapped(name, action) for name, action in registry.items()}
        result = GraphPolicyExecutor(policy, registry).execute({"seed": seed,
                                                               "target_id": "alphabet_soup"})
        values = backend.env.env.containment()
        success = backend.task_success()
        backend.capture().save(output / "final")
        if hashlib.sha256(graph.read_bytes()).digest() != hashlib.sha256(graph_bytes).digest():
            raise RuntimeError("the original graph changed during execution")
        return {"ok": result.status == "succeeded" and success and all(checkpoints.values()),
            **metadata, "graph": result.as_dict(), "graph_path": str(graph),
            "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
            "graph_modified": False, "plan_sha256": hashlib.sha256(plan_text.encode()).hexdigest(),
            "steps": backend.steps, "elapsed_sim_s": backend.steps / 20,
            "initial_state": initial_state,
            "trajectory": trajectory, "evaluation": {"status": "succeeded" if success else "failed",
                "checkpoints": checkpoints, "collision_check": "robot vs floor objects and basket",
                "disallowed_collisions": backend.disallowed_collisions,
                "task_success": success, **values},
            "comparison_limits": ["oracle scene geometry rather than saved perception",
                                  "different robot meshes controller contacts and solver",
                                  "contact abort uses geometry pairs not Isaac impulse threshold",
                                  "custom LIBERO task not an official suite score"]}
    finally:
        backend.close()
