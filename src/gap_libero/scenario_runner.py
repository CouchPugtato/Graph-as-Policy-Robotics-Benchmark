"""Bounded local Qwen execution for custom LIBERO task families."""

import hashlib
import importlib
import math
from functools import partial
from pathlib import Path

import numpy as np

from gap_isaac.graph_policy import (
    GraphExecutionBoundExceeded,
    GraphPolicy,
    GraphPolicyExecutor,
    NodeResult,
)
from gap_libero.actions import LiberoGraphActions, RunPlan
from gap_libero.backend import LiberoBackend
from gap_libero.grocery_comparison import closest_parallel_jaw_yaw
from gap_libero.local_vlm import LocalVLM, renderer_headroom
from gap_libero.scenario_geometry import estimate
from gap_libero.scenario_specs import (
    DESCRIPTIONS,
    PACKING_BASKET_SIZE,
    PRIORS,
    packing_items,
    validate_case,
)
from gap_libero.vision import VisionActions

ROOT = Path(__file__).resolve().parents[2]


def create_backend(scenario, seed, variation, enable_motion):
    from gap_libero.runtime_config import prepare_runtime_config

    prepare_runtime_config()
    from libero.libero.envs import OffScreenRenderEnv

    importlib.import_module("gap_libero.grocery_task")
    env = OffScreenRenderEnv(
        bddl_file_name=str(ROOT / "libero/grocery.bddl"),
        scene_seed=seed,
        scenario=scenario,
        variation=variation,
        robots=["OnTheGroundPanda"],
        initialization_noise=None,
        camera_names=["agentview"],
        camera_widths=640,
        camera_heights=480,
        camera_depths=True,
        controller="OSC_POSE",
        control_freq=20,
        horizon=4050,
        hard_reset=False,
        ignore_done=False,
        render_gpu_device_id=0,
    )
    try:
        env.seed(seed)
        env.reset()
        for _ in range(40):
            observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        backend = LiberoBackend(
            env,
            motion_enabled=enable_motion,
            max_steps=4000,
            episode_step_limit=4000,
            workspace=[[-0.35, -0.85, 0.02], [1, 0.85, 1.1]],
            forbidden_geom_pairs=env.env.forbidden_pairs(),
        )
        backend.observation = observation
        backend.check_workspace(backend.eef_position())
        backend.check_contacts()
        return backend
    except BaseException:
        env.close()
        raise


class SparseRecorder:
    def __init__(self, backend, directory):
        from gap_libero.recording import FrameRecorder

        self.recorder = FrameRecorder(directory)
        self.latest = None
        self.stride = math.ceil(backend.max_steps / 999)

    def record(self, observation, step):
        self.latest = (observation, step)
        if step % self.stride == 0:
            self.recorder.record(observation, step)

    def close(self):
        if self.latest and self.latest[1] != self.recorder.frames[-1]["step"]:
            self.recorder.record(*self.latest)
        self.recorder.close()


class ScenarioActions:
    def __init__(self, backend, scenario, seed, output):
        self.backend, self.scenario, self.seed, self.output = backend, scenario, seed, output
        self.requested = (
            packing_items(seed)
            if scenario == "packing"
            else (backend.env.env.scene_plan.target_id,)
        )
        self.completed = []
        self.index = 0
        self.target = self.requested[0]
        self.goal = "burner" if scenario == "popcorn" else "basket"
        self.phase = "transfer"
        self.vision = self.bindings = None
        self.local_calls = 0
        self.maximum_local_calls = 8
        self.cycle = 0
        self.checkpoints = {
            "target_grasped": False,
            "target_lifted": False,
            "target_released": False,
            "target_contained": False,
        }
        self.trajectory = []

    def diagnostics(self):
        env = self.backend.env
        data, model = env.sim.data, env.sim.model
        return {
            "eef_position_m": self.backend.eef_position().tolist(),
            "arm_joints": env.robots[0]._joint_positions.tolist(),
            "eef_orientation_xyzw": self.backend.eef_orientation().tolist(),
            "objects": {
                name: {
                    "position_m": data.get_body_xpos(obj.root_body).tolist(),
                    "rotation": data.get_body_xmat(obj.root_body).tolist(),
                }
                for name, obj in env.env.objects_dict.items()
            },
            "gripper_joints": [
                float(data.get_joint_qpos(name)) for name in env.robots[0].gripper.joints
            ],
            "contacts": [
                [model.geom_id2name(c.geom1), model.geom_id2name(c.geom2)]
                for c in data.contact[: data.ncon]
            ],
        }

    def set_target(self, target, phase="transfer"):
        if self.vision:
            self.local_calls += self.vision.identifier.calls
        self.target, self.phase = target, phase
        self.vision = self.bindings = None
        self.cycle += 1
        self.backend.env.env.active_target = target
        self.backend.forbidden_pairs = {
            frozenset(self.backend.env.sim.model.geom_name2id(name) for name in pair)
            for pair in self.backend.env.env.forbidden_pairs()
        }

    def select_next(self, state):
        if self.index >= len(self.requested):
            return NodeResult("false", "all requested items have been visited")
        self.set_target(self.requested[self.index])
        self.index += 1
        return NodeResult(
            "true",
            outputs={"target_id": self.target, "target_description": DESCRIPTIONS[self.target]},
        )

    def observe(self, state):
        if self.vision:
            self.local_calls += self.vision.identifier.calls
        if self.local_calls >= self.maximum_local_calls:
            raise RuntimeError("local vision episode call budget exhausted")
        identifier = LocalVLM(max_calls=min(2, self.maximum_local_calls - self.local_calls))
        self.vision = VisionActions(
            self.backend.capture,
            self.output / f"perception-{self.cycle:02d}",
            identifier=identifier,
            descriptions=DESCRIPTIONS,
            geometry_estimator=partial(estimate, basket_size=PACKING_BASKET_SIZE)
            if self.scenario == "packing"
            else estimate,
            silhouette_objects=set(PRIORS) - {"basket"},
        )
        self.cycle += 1
        return self.vision.observe(state)

    def perceive(self, verb, container, state):
        if self.vision is None:
            raise RuntimeError("capture before using perception tools")
        name = self.goal if container else self.target
        function = {
            "detect": self.vision.detect,
            "identify": self.vision.identify,
            "segment": self.vision.segment,
            "estimate": self.vision.geometry,
            "check": self.vision.found,
        }[verb]
        return function(name, state)

    def plan(self):
        if self.vision is None or not self.vision.geometries.get(self.target, {}).get("valid"):
            raise RuntimeError("verified local vision target geometry is required")
        geometry = self.vision.geometries[self.target]
        position = list(geometry["position_m"])
        if self.target == "pan":
            handle_yaw = closest_parallel_jaw_yaw(geometry["yaw_rad"], 0)
            position[:2] = (
                np.asarray(position[:2])
                - 0.035 * np.array([math.cos(handle_yaw), math.sin(handle_yaw)])
            ).tolist()
        shape, size = PRIORS[self.target]
        yaw = (
            geometry["yaw_rad"] + math.pi / 2
            if shape == "box"
            else math.atan2(position[1], position[0]) + math.pi / 2
        )
        yaw = closest_parallel_jaw_yaw(yaw, math.atan2(position[1], position[0]) + math.pi / 2)
        orientation = [math.cos(yaw / 2), math.sin(yaw / 2), 0, 0]

        def move(p, hold=False, quaternion=None):
            return {
                "kind": "move",
                "position_m": list(map(float, p)),
                "orientation_xyzw": quaternion or orientation,
                "max_steps": 400,
                "tolerance_m": 0.004,
                "maximum_translation_command": 0.15,
                "require_grasp": hold,
            }

        grasp_z = max(
            0.025,
            position[2]
            + (
                0.035
                if self.target in ("alphabet_soup", "tomato_sauce", "chocolate_pudding")
                else 0.005
                if self.target == "cream_cheese"
                else min(0.025, size[2] * 0.2)
            ),
        )
        transit = 0.34 if self.scenario == "popcorn" else 0.28
        actions = {
            "open_gripper": {"kind": "open", "max_steps": 25},
            "align_over_target": move([*position[:2], transit]),
            "descend_to_grasp": move([*position[:2], grasp_z]),
            "close_gripper": {"kind": "close", "max_steps": 80, "preload_steps": 8},
            "lift_grasped": move([*position[:2], transit], True),
            "release_object": {"kind": "open", "max_steps": 45},
            "settle_release": {"kind": "verify", "max_steps": 40},
            "return_ready": move([0.30, 0, 0.40]),
        }
        if self.phase == "transfer":
            if not self.vision.geometries.get(self.goal, {}).get("valid"):
                raise RuntimeError("verified local vision destination geometry is required")
            goal = self.vision.geometries[self.goal]
            xy = np.array(goal["position_m"][:2])
            release_z = 0.145
            destination_orientation = orientation
            if self.scenario == "fulfillment":
                destination_yaw = closest_parallel_jaw_yaw(
                    goal["yaw_rad"], math.atan2(xy[1], xy[0]) + math.pi / 2
                )
                destination_orientation = [
                    math.cos(destination_yaw / 2),
                    math.sin(destination_yaw / 2),
                    0,
                    0,
                ]
                release_z = 0.115 if shape == "cylinder" else 0.145
            if self.scenario == "packing":
                basket_yaw = goal["yaw_rad"]
                direction = -1 if len(self.completed) == 0 else 1
                local_offset = direction * np.array([0.055, 0])
                basket_rotation = np.array(
                    [
                        [math.cos(basket_yaw), -math.sin(basket_yaw)],
                        [math.sin(basket_yaw), math.cos(basket_yaw)],
                    ]
                )
                xy += basket_rotation @ local_offset
                # open across the basket and choose the nearest equivalent jaw orientation
                desired_yaw = closest_parallel_jaw_yaw(
                    basket_yaw + math.pi / 2, yaw
                )
                destination_orientation = [
                    math.cos(desired_yaw / 2),
                    math.sin(desired_yaw / 2),
                    0,
                    0,
                ]
                release_z = max(0.105, 0.02 + size[2] / 2 + (grasp_z - position[2]) + 0.008)
            elif self.scenario == "popcorn":
                destination_yaw = math.atan2(xy[1], xy[0]) + math.pi / 2
                handle_yaw = destination_yaw - math.pi / 2
                destination_orientation = [
                    math.cos(destination_yaw / 2),
                    math.sin(destination_yaw / 2),
                    0,
                    0,
                ]
                xy += 0.085 * np.array([math.cos(handle_yaw), math.sin(handle_yaw)])
                release_z = goal["position_m"][2] + 0.005 + 0.018 + 0.014 + 0.025
            actions.update(
                move_over_basket=move([*xy, transit], True, destination_orientation),
                position_for_release=move([*xy, release_z], True, destination_orientation),
                retract_from_basket=move([*xy, transit], False, destination_orientation),
            )
            actions["position_for_release"].update(tolerance_m=0.0015, position_gain=4)
        else:
            turn = 0.8 if self.phase == "on" else -0.8
            final_yaw = yaw + turn
            actions["turn_stove_" + self.phase] = move(
                [*position[:2], grasp_z],
                True,
                [math.cos(final_yaw / 2), math.sin(final_yaw / 2), 0, 0],
            )
            actions["retract_from_basket"] = move([*position[:2], transit])
        return RunPlan.model_validate(
            {
                "suite": "grocery_ground",
                "task_id": 0,
                "seed": self.seed,
                "target_object": self.target,
                "workspace_m": self.backend.workspace.tolist(),
                "max_episode_steps": 1000,
                "grasp_reference_offset_m": [0.085, 0, 0.014]
                if self.target == "pan"
                else [0, 0, 0],
                "maximum_grasp_rotation_rad": 0.25 if self.target == "pan" else None,
                "actions": actions,
            }
        )

    def motion(self, name, state):
        if self.bindings is None:
            plan = self.plan()
            path = self.output / f"plan-{self.cycle:02d}.json"
            path.write_text(plan.model_dump_json(indent=2))
            self.bindings = LiberoGraphActions(self.backend, plan)
        if name == "release_object" and self.scenario == "packing":
            env = self.backend.env
            gripper = env.robots[0].gripper
            data, model = env.sim.data, env.sim.model
            target_geoms = {
                model.geom_name2id(geom) for geom in env.env.objects_dict[self.target].contact_geoms
            }
            finger_geoms = {model.geom_name2id(geom) for geom in gripper.contact_geoms}
            self.bindings.gripper = 0
            clear = 0
            # keep opening after contact clears before retracting the hand
            for _ in range(45):
                commands = []
                for index, joint in enumerate(gripper.joints):
                    low, high = model.jnt_range[model.joint_name2id(joint)]
                    position = float(data.get_joint_qpos(joint))
                    desired = np.clip(position + (0.004 if index == 0 else -0.004), low, high)
                    commands.append(2 * (desired - low) / (high - low) - 1)
                gripper.current_action = np.asarray(commands)
                self.bindings.command()
                touching = any(
                    (c.geom1 in target_geoms and c.geom2 in finger_geoms)
                    or (c.geom2 in target_geoms and c.geom1 in finger_geoms)
                    for c in data.contact[: data.ncon]
                )
                clear = clear + 1 if not touching else 0
                opened = all(
                    abs(float(data.get_joint_qpos(joint))) >= 0.038
                    for joint in gripper.joints
                )
                if clear >= 3 and opened:
                    self.bindings.hold_reference = self.bindings.hold_rotation = None
                    return NodeResult(detail="target released after full aperture and clear contacts")
            return NodeResult("failed", "release controller could not open fully and clear target contacts")
        if name == "settle_release":
            stable = 0
            for _ in range(120):
                self.bindings.command()
                values = self.backend.env.env.containment()
                ready = (
                    (
                        values["pan_on_burner"]
                        and values["pan_released"]
                        and values["pan_speed_m_s"] <= 0.1
                    )
                    if self.scenario == "popcorn"
                    else (
                        values["target_contained"]
                        and values["target_released"]
                        and values["target_speed_m_s"] <= 0.1
                    )
                )
                stable = stable + 1 if ready else 0
                if stable >= 10:
                    return NodeResult(detail="release stable for ten consecutive samples")
            key = "pan_on_burner" if self.scenario == "popcorn" else "target_contained"
            reason = (
                "placement outside destination bounds"
                if not values[key]
                else "release contact remains"
                if not values["pan_released" if self.scenario == "popcorn" else "target_released"]
                else "released object did not become stationary"
            )
            return NodeResult("failed", reason + "; bounded settle verification failed", outputs=values)
        result = self.bindings.timed_action(name, state)
        if result.outcome == "succeeded":
            if name == "close_gripper":
                self.checkpoints["target_grasped"] = self.backend.grasped(self.target)
            if name == "lift_grasped":
                self.checkpoints["target_lifted"] = self.backend.grasped(self.target)
            if name.startswith("turn_stove_"):
                values = self.backend.env.env.containment()
                valid = values["stove_on" if self.phase == "on" else "stove_off"]
                if not valid:
                    return NodeResult("failed", "physical knob did not reach the required range")
                self.backend.env.env.popcorn_history[
                    "stove_turned_" + ("on" if self.phase == "on" else "off")
                ] = True
        return result

    def placement(self, state):
        values = self.backend.env.env.containment()
        if self.scenario == "popcorn":
            valid = (
                values["pan_on_burner"]
                and values["pan_released"]
                and values["pan_speed_m_s"] <= 0.1
            )
            self.backend.env.env.popcorn_history["pan_placed"] = valid
        else:
            valid = (
                values["target_contained"]
                and values["target_released"]
                and values["target_speed_m_s"] <= 0.1
            )
        if valid and self.target not in self.completed:
            self.completed.append(self.target)
        self.checkpoints.update(
            target_released=bool(values.get("target_released", values.get("pan_released"))),
            target_contained=bool(valid),
        )
        return NodeResult(
            "true" if valid else "false",
            outputs={"placement_verified": bool(valid), "completed_items": list(self.completed)},
        )

    def wait_heating(self, state):
        env = self.backend.env.env
        for _ in range(40):
            values = env.containment()
            if not values["stove_on"] or not values["pan_on_burner"] or not values["pan_released"]:
                return NodeResult("failed", "pan and stove conditions were not maintained")
            self.bindings.command()
            env.popcorn_history["heating_steps"] += 1
        return NodeResult(detail="bounded heating abstraction completed without thermal simulation")

    def packing_evidence(self):
        return {name: self.backend.env.env.containment(name) for name in self.requested}

    def check_all_packed(self, state):
        items = self.packing_evidence()
        missing = [
            name
            for name, values in items.items()
            if not (
                values["target_contained"]
                and values["target_released"]
                and values["target_speed_m_s"] <= 0.1
            )
        ]
        return NodeResult(
            "false" if missing else "true",
            "unpacked items: " + ", ".join(missing) if missing else "all items verified",
            outputs={"items": items},
        )

    def registry(self):
        actions = {
            "observe_scene": self.observe,
            "select_next_item": self.select_next,
            "check_grasp": lambda state: NodeResult(
                "true" if self.backend.grasped(self.target) else "false"
            ),
            "check_placement": self.placement,
            "check_all_packed": self.check_all_packed,
            "check_popcorn_complete": lambda state: NodeResult(
                "true" if self.backend.task_success() else "false"
            ),
            "wait_heating": self.wait_heating,
        }
        for phase in ("on", "off"):

            def select(state, phase=phase):
                self.set_target("knob", phase)
                return NodeResult(outputs={"target_id": "knob"})

            actions["select_knob_" + phase] = select
        for prefix in ("target", "container"):
            for verb in ("detect", "identify", "segment", "estimate", "check"):
                actions[f"{verb}_{prefix}"] = lambda state, v=verb, c=prefix == "container": (
                    self.perceive(v, c, state)
                )
        for name in (
            "open_gripper",
            "align_over_target",
            "descend_to_grasp",
            "close_gripper",
            "lift_grasped",
            "move_over_basket",
            "position_for_release",
            "release_object",
            "retract_from_basket",
            "settle_release",
            "return_ready",
            "turn_stove_on",
            "turn_stove_off",
        ):
            actions[name] = lambda state, n=name: self.motion(n, state)
        return actions


def run(output, graph, scenario, seed=0, variation="mixed", enable_motion=False, record=False):
    validate_case(scenario, seed, variation)
    graph_bytes = graph.read_bytes()
    policy = GraphPolicy.from_json(graph)
    settings = LocalVLM().preflight()
    settings.update(renderer_headroom())
    backend = create_backend(scenario, seed, variation, enable_motion)
    try:
        backend.capture().save(output / "initial")
        metadata = {
            "backend": "libero",
            "scenario": scenario,
            "seed": seed,
            "variation": variation,
            "basket_size_m": list(PACKING_BASKET_SIZE)
            if scenario == "packing"
            else None
            if scenario == "popcorn"
            else list(PRIORS["basket"][1]),
            "custom_task": True,
            "official_libero_benchmark": False,
            "local_vlm": settings,
            "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
            "implementation_files_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in [
                    *sorted((ROOT / "src/gap_libero").glob("*.py")),
                    ROOT / "src/gap_isaac/evaluate.py",
                    ROOT / "src/gap_isaac/grocery_scene.py",
                    ROOT / "configs/tasks/grocery_fulfillment.yaml",
                    ROOT / "libero/grocery.bddl",
                ]
            },
            "models_loaded_by_worker": False,
            "thermal_simulation": False,
        }
        if not enable_motion:
            return {"ok": True, **metadata, "motion_enabled": False, "execution_validated": False}
        if backend.task_success():
            raise RuntimeError("scenario is already successful at reset")
        if record:
            backend.recorder = SparseRecorder(backend, output / "recording")
            backend.recorder.record(backend.observation, 0)
        actions = ScenarioActions(backend, scenario, seed, output)
        registry = actions.registry()
        trajectory = []

        def wrap(name, action):
            def call(state):
                start = backend.steps
                before = actions.diagnostics()
                print(f"{scenario} | {actions.target} | {name} | step {start}/4000", flush=True)
                try:
                    result = action(state)
                except Exception as error:
                    result = NodeResult("failed", f"{type(error).__name__}: {error}")
                if result.outcome == "failed" or (
                    result.outcome == "false" and name not in ("select_next_item",)
                ):
                    domain = (
                        "geometry"
                        if name.startswith(("estimate_", "check_target", "check_container"))
                        else "perception"
                        if name.startswith(("observe_", "detect_", "identify_", "segment_"))
                        else "placement"
                        if name in ("settle_release", "check_placement", "check_all_packed")
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
                        "outcome": result.outcome,
                        "start_step": start,
                        "step": backend.steps,
                        "eef_position_m": backend.eef_position().tolist(),
                        "target_grasped": backend.grasped(actions.target),
                        "before": before,
                        "after": actions.diagnostics(),
                    }
                )
                return result

            return call

        bound_exceeded = False
        try:
            result = GraphPolicyExecutor(
                policy, {name: wrap(name, fn) for name, fn in registry.items()}, max_steps=160
            ).execute({"seed": seed})
        except GraphExecutionBoundExceeded as error:
            result = error.result
            bound_exceeded = True
        valid = backend.task_success()
        calls = actions.local_calls + (actions.vision.identifier.calls if actions.vision else 0)
        backend.capture().save(output / "final")
        if graph.read_bytes() != graph_bytes:
            raise RuntimeError("graph changed during scenario execution")
        evidence = backend.env.env.containment()
        if scenario == "packing":
            evidence["items"] = actions.packing_evidence()
            evidence["all_requested_contained"] = all(
                values["target_contained"] for values in evidence["items"].values()
            )
            evidence["all_requested_released"] = all(
                values["target_released"] for values in evidence["items"].values()
            )
        graph_result = result.as_dict()
        if bound_exceeded:
            graph_result["execution_bound_exceeded"] = True
        return {
            "ok": result.status == "succeeded" and valid,
            **metadata,
            "graph": graph_result,
            "steps": backend.steps,
            "trajectory": trajectory,
            "completed_items": actions.completed,
            "perception": {
                "backend": "local-vlm",
                "local_vlm_calls": calls,
                "local_vlm_model": settings["model"],
                "uses_oracle_object_poses": False,
            },
            "evaluation": {
                "task_success": valid,
                "status": "succeeded" if valid else "failed",
                "checkpoints": actions.checkpoints,
                "disallowed_collisions": backend.disallowed_collisions,
                "scenario_evidence": evidence,
            },
            "recording": {"frames": "recording/frames.json", "fps": 20} if record else None,
        }
    finally:
        backend.close()
