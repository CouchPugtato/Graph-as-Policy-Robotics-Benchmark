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
MOTION_ACTIONS = ("open_gripper", "align_over_target", "descend_to_grasp", "close_gripper",
                  "lift_grasped", "move_over_basket", "position_for_release", "release_object",
                  "retract_from_basket", "settle_release")


def closest_parallel_jaw_yaw(yaw: float, reference: float = 0.0) -> float:
    if not math.isfinite(yaw) or not math.isfinite(reference):
        raise ValueError("jaw orientation angles must be finite")
    # swapping the two identical jaws leaves the basket alignment unchanged
    difference = (yaw - reference + math.pi / 2) % math.pi - math.pi / 2
    return reference + difference


def create_grocery_backend(seed=0, motion_enabled=False, max_steps=1000, workspace=None,
                           camera_size=(320, 240), **settings):
    from gap_libero.runtime_config import prepare_runtime_config

    prepare_runtime_config()
    from libero.libero.envs import OffScreenRenderEnv

    from gap_libero import grocery_task  # noqa: F401 - registers the custom libero problem

    width, height = camera_size
    if not 1 <= width <= 640 or not 1 <= height <= 480:
        raise ValueError("camera resolution exceeds the RGB-D observation budget")
    env = OffScreenRenderEnv(bddl_file_name=str(ROOT / "libero/grocery.bddl"),
        scene_seed=seed, robots=["OnTheGroundPanda"], initialization_noise=None,
        camera_names=["agentview"], camera_widths=width, camera_heights=height, camera_depths=True,
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


def build_action_plan(backend, seed, perceived=None, *, native=False):
    if native and perceived is None:
        raise ValueError("native motion requires image derived geometry")
    if perceived is None:
        env = backend.env.env
        data = env.sim.data
        target = data.get_body_xpos(env.objects_dict["alphabet_soup"].root_body).copy()
        basket = data.get_body_xpos(env.objects_dict["basket"].root_body).copy()
        basket_rotation = data.get_body_xmat(env.objects_dict["basket"].root_body)
        basket_yaw = math.atan2(basket_rotation[1, 0], basket_rotation[0, 0])
    else:
        target = perceived["alphabet_soup"]["position_m"]
        basket = perceived["basket"]["position_m"]
        basket_yaw = perceived["basket"]["yaw_rad"]
    grasp_z = float(target[2] + (0.025 if native else 0.035))
    transit_z = 0.28
    # keep the wrist near its neutral heading as the arm moves around the base
    grasp_yaw = math.atan2(float(target[1]), float(target[0])) + math.pi / 2
    orientation = [math.cos(grasp_yaw / 2), math.sin(grasp_yaw / 2), 0, 0]
    approach_heading = math.atan2(float(basket[1]), float(basket[0])) + math.pi / 2
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
    if native:
        plan.update(suite="libero_object", init_state_id=seed)
        plan["actions"]["position_for_release"]["position_m"][2] = 0.24
        orientation = [1.0, 0.0, 0.0, 0.0]
        for action in plan["actions"].values():
            if action["kind"] == "move":
                action["orientation_xyzw"] = orientation
    return RunPlan.model_validate(plan)


def native_evidence(backend):
    data = backend.env.sim.data
    target = backend.target_object("alphabet_soup")
    position = data.get_body_xpos(target.root_body).tolist()
    success = backend.task_success()
    return {"target_position_m": position,
            "target_released": not backend.grasped("alphabet_soup"),
            "target_contained": success, "official_goal_satisfied": success}


def compare_grocery(output: Path, graph: Path = DEFAULT_GRAPH, seed=0, capture_only=False,
                    record=False, perception_backend=None, native=False):
    graph_bytes = graph.read_bytes()
    policy = GraphPolicy.from_json(graph)
    vision_backend = perception_backend or policy.metadata.get("perception_backend")
    if native and (not 0 <= seed < 10 or vision_backend != "local-vlm"):
        raise ValueError("native object task requires local Qwen and initial states zero through nine")
    if vision_backend not in (None, "rgbd-color", "local-vlm"):
        raise ValueError("unsupported perception backend no oracle fallback is allowed")
    identifier, vlm_settings = None, None
    if vision_backend == "local-vlm":
        from gap_libero.local_vlm import LocalVLM, renderer_headroom

        identifier = LocalVLM()
        vlm_settings = identifier.preflight()
        vlm_settings.update(renderer_headroom())
    if native:
        from gap_libero.backend import create_backend
        from gap_libero.benchmarks.runtime import activate

        selected = activate("libero-object")
        backend, metadata = create_backend(
            suite="libero_object", task_id=0, init_state_id=seed, seed=seed,
            camera_size=(640, 480), motion_enabled=not capture_only, max_steps=1000,
            workspace=[[-0.8, -0.8, 0.015], [0.8, 0.8, 1.2]],
            object_aliases={"alphabet_soup": "alphabet_soup_1", "basket": "basket_1"},
            verify_after_goal=True)
        metadata.update(custom_task=False, official_libero_benchmark=True,
                        source_revision=selected["revision"],
                        post_goal_verification=True,
                        benchmark_scope="libero-object-task-0-initial-states-0-9")
        model = backend.env.sim.model
        obstacle_geoms = {geom for obj in backend.env.env.objects_dict.values()
                          for geom in obj.contact_geoms}
        obstacle_geoms.update(name for name in model.geom_names if "floor" in name)
        robot_geoms = [name for name in model.geom_names
                       if name.startswith("robot0_") and "link0" not in name and "link1" not in name]
        robot_geoms += [name for name in model.geom_names
                        if name.startswith("gripper0_") and "finger" not in name]
        backend.forbidden_pairs = {frozenset((model.geom_name2id(a), model.geom_name2id(b)))
                                   for a in robot_geoms for b in obstacle_geoms}
        non_target = obstacle_geoms - set(backend.target_object("alphabet_soup").contact_geoms)
        backend.forbidden_pairs.update(
            frozenset((model.geom_name2id(a), model.geom_name2id(b)))
            for a in model.geom_names if a.startswith("gripper0_") and "finger" in a
            for b in non_target)
        backend.check_contacts()
        if not capture_only:
            for _ in range(20):
                backend.step([0, 0, 0, 0, 0, 0, -1])
            metadata["settle_steps"] = 20
    else:
        backend, metadata = create_grocery_backend(seed, motion_enabled=not capture_only,
                                                  camera_size=(640, 480) if vision_backend else (320, 240))
    evidence = lambda: native_evidence(backend) if native else backend.env.env.containment()
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
        if vision_backend:
            name = "src/gap_libero/vision.py"
            metadata["implementation_files_sha256"][name] = hashlib.sha256(
                (ROOT / name).read_bytes()).hexdigest()
            metadata["geometry_source"] = "live-rgbd-color-and-shape-priors"
            metadata["perception_backend"] = vision_backend
            if identifier:
                for name in ("src/gap_libero/local_vlm.py", "src/gap_isaac/local_vision_smoke.py"):
                    metadata["implementation_files_sha256"][name] = hashlib.sha256(
                        (ROOT / name).read_bytes()).hexdigest()
                metadata.update(local_vlm=vlm_settings, models_loaded=True,
                                models_loaded_by_worker=False,
                                geometry_source="live-rgbd-local-vlm-and-shape-priors")
        metadata["containment_evaluator"] = ("libero.env.check_success" if native else
                                             "gap_isaac.evaluate.target_is_contained")
        metadata["contact_parameters"] = {"proxy_margin_m": 0, "proxy_solref": [0.01, 1.0],
                                         "finger_force_limit_n": 7.2}
        (output / "scene.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        backend.capture().save(output / "initial")
        if capture_only:
            return {"ok": True, **metadata, "motion_enabled": False,
                    "initial_observation": str(output / "initial")}
        plan_text, bindings = None, None
        vision = None
        if vision_backend:
            from gap_libero.vision import VisionActions

            from gap_libero.vision import native_candidates, native_geometry

            options = ({"candidate_provider": native_candidates, "geometry_estimator": native_geometry,
                        "descriptions": {"alphabet_soup": "the ALPHABET SOUP metal can with a blue upper label and orange soup with pasta letters below a white band not the red tomato sauce can",
                                         "basket": "the open woven wicker basket"}} if native else {})
            vision = VisionActions(backend.capture, output / "perception", identifier=identifier, **options)

        def motion(name, state):
            nonlocal bindings, plan_text
            if bindings is None:
                if vision is not None and not all(
                        vision.geometries.get(key, {}).get("valid") is True
                        and state.get(key + "_found") is True
                        for key in ("alphabet_soup", "basket")):
                    raise RuntimeError("image derived target and container geometry required")
                plan = build_action_plan(backend, seed, vision.geometries if vision else None, native=native)
                plan_text = plan.model_dump_json(indent=2)
                (output / "plan.json").write_text(plan_text, encoding="utf-8")
                bindings = LiberoGraphActions(backend, plan)
            return bindings.timed_action(name, state)

        registry = {name: (lambda state, key=name: motion(key, state)) for name in MOTION_ACTIONS}
        if vision:
            registry.update(vision.registry())

        def predicate(name, value):
            return NodeResult("true" if value else "false", outputs={name: bool(value)})

        registry["check_grasp"] = lambda state: predicate(
            "grasp_verified", bool(bindings and bindings.hold_reference is not None
                                   and backend.grasped("alphabet_soup")))
        registry["check_placement"] = lambda state: predicate(
            "placement_verified", backend.task_success())
        checkpoints = {"target_grasped": False, "target_lifted": False,
                       "target_released": False, "target_contained": False}
        trajectory = []
        initial_height = float(evidence()["target_position_m"][2])

        def phase_state():
            data, model = backend.env.sim.data, backend.env.sim.model
            return {"step": backend.steps,
                "eef_position_m": backend.eef_position().tolist(),
                "eef_orientation_xyzw": backend.eef_orientation().tolist(),
                "arm_joint_positions_rad": backend.env.robots[0]._joint_positions.tolist(),
                "grasped": backend.grasped("alphabet_soup"),
                "objects": {name: {
                    "position_m": data.get_body_xpos(obj.root_body).tolist(),
                    "rotation_matrix": data.get_body_xmat(obj.root_body).tolist(),
                    "linear_velocity_m_s": data.get_body_xvelp(obj.root_body).tolist()}
                    for name, obj in backend.env.env.objects_dict.items()},
                "contacts": [{"geometries": [model.geom_id2name(contact.geom1),
                                              model.geom_id2name(contact.geom2)],
                              "distance_m": float(contact.dist)}
                             for contact in data.contact[:min(data.ncon, 64) if native else data.ncon]],
                "contact_count": data.ncon,
                **evidence()}

        initial_state = phase_state()

        def wrapped(name, action):
            def execute(state):
                print(f"LIBERO graph action: {name}", flush=True)
                before = phase_state()
                try:
                    if name == "settle_release" and bindings is not None:
                        for _ in range(40):
                            bindings.command()
                    result = action(state)
                except Exception as error:
                    result = NodeResult("failed", f"{type(error).__name__}: {error}")
                if native and result.outcome in ("failed", "false"):
                    domain = ("geometry" if name.startswith(("estimate_", "check_target", "check_container"))
                              else "perception" if name.startswith(("observe_", "detect_", "identify_", "segment_"))
                              else "placement" if name in ("settle_release", "check_placement")
                              else "control")
                    result = NodeResult(result.outcome, result.detail,
                                        outputs={**(result.outputs or {}), "failure_domain": domain})
                values = evidence()
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
        values = evidence()
        success = backend.task_success()
        backend.capture().save(output / "final")
        if hashlib.sha256(graph.read_bytes()).digest() != hashlib.sha256(graph_bytes).digest():
            raise RuntimeError("the original graph changed during execution")
        return {"ok": result.status == "succeeded" and success and all(checkpoints.values()),
            **metadata, "graph": result.as_dict(), "graph_path": str(graph),
            "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
            "graph_modified": False, "plan_sha256": (
                hashlib.sha256(plan_text.encode()).hexdigest() if plan_text else None),
            "steps": backend.steps, "elapsed_sim_s": backend.steps / 20,
            "initial_state": initial_state,
            "perception": {"backend": vision_backend,
                "target_found": result.state.get("alphabet_soup_found") is True,
                "container_found": result.state.get("basket_found") is True,
                "target_identified": result.state.get("alphabet_soup_identified") is True,
                "container_identified": result.state.get("basket_identified") is True,
                "local_vlm_calls": identifier.calls if identifier else 0,
                "local_vlm_model": identifier.model if identifier else None,
                "uses_oracle_object_poses": vision is None},
            "trajectory": trajectory, "evaluation": {"status": "succeeded" if success else "failed",
                "checkpoints": checkpoints, "collision_check": "robot vs floor objects and basket",
                "disallowed_collisions": backend.disallowed_collisions,
                "task_success": success, **values},
            "comparison_limits": [(
                ("local VLM identification over RGB-D foreground regions not DINO or SAM" if native else
                 "local VLM identification over color region proposals not DINO or SAM")
                if identifier else "color segmentation and known shape priors "
                "rather than neural object identification" if vision else
                "oracle scene geometry rather than saved perception"),
                                  "different robot meshes controller contacts and solver",
                                  "contact abort uses geometry pairs not Isaac impulse threshold",
                                  ("native LIBERO Object task zero only not a whole suite score" if native else
                                   "custom LIBERO task not an official suite score")]}
    finally:
        backend.close()
