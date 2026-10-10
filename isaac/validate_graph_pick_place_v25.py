"""Execute one bounded Graph-as-Policy pick-and-place smoke in native Isaac Sim."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

RUN_REVISION = "graph-ground-perceived-pick-place-v25-alignment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", default="alphabet_soup")
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--baseline-diagnostics", action="store_true")
    parser.add_argument("--physics-hz", type=int, choices=(60, 120), default=120)
    parser.add_argument("--target-geometry", type=Path)
    parser.add_argument("--basket-geometry", type=Path)
    parser.add_argument("--grasp-result", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect-fingers", action="store_true")
    mode.add_argument("--grasp-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    global RUN_REVISION
    args = parse_args()
    if args.baseline_diagnostics:
        raise SystemExit("v25 cannot select the v23 geometry adapter")
    if args.grasp_result:
        receipt = json.loads(args.grasp_result.read_text())
        if receipt.get("run_revision") != RUN_REVISION:
            raise SystemExit("v25 graph requires a matching v25 grasp receipt")
    perceived = args.target_geometry is not None or args.basket_geometry is not None
    if perceived and (args.inspect_fingers or not args.target_geometry or not args.basket_geometry):
        raise SystemExit("Perceived geometry requires both files and cannot use inspection mode")
    if perceived and not args.grasp_validation and not args.grasp_result:
        raise SystemExit("Perceived graph requires its successful bounded grasp result")
    print(f"Graph run revision: {RUN_REVISION}", flush=True)
    for label, path in (("USD stage", args.stage), ("task", args.task), ("graph", args.graph)):
        if not path.is_file():
            raise SystemExit(f"{label} does not exist: {path}")

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    # Validate generated artifacts before importing/launching Isaac or allocating GPU memory.
    from gap_isaac.graph_policy import GraphPolicy
    from gap_isaac.policy_generation import AUTHORING, sha256, verify_accepted

    graph = GraphPolicy.from_json(args.graph)
    generation_receipt = None
    reference_path = project_root / "graphs/reference/grocery_single_target.json"
    if args.graph.resolve() != reference_path.resolve():
        if not perceived or args.grasp_validation or args.inspect_fingers:
            raise SystemExit("Generated graph requires perceived geometry and Graph mode")
        generation_receipt = verify_accepted(args.graph)
    graph_hash = sha256(args.graph.read_bytes())
    graph_authoring = (
        AUTHORING if generation_receipt else "hand-written-reference; not model-generated"
    )

    from isaacsim import SimulationApp

    diagnostics = {}
    descent_trace_stream = None

    def write_result(payload: dict) -> None:
        payload["run_revision"] = RUN_REVISION
        payload.setdefault("diagnostics", diagnostics)
        if args.result_file is not None:
            args.result_file.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "width": 640,
            "height": 480,
            "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False, "limit_cpu_threads": 2,
            "extra_args": ["--enable", "isaacsim.robot_motion.motion_generation"],
        }
    )
    try:
        import numpy as np
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_motion.motion_generation import (
            ArticulationKinematicsSolver,
            LulaKinematicsSolver,
            load_supported_lula_kinematics_solver_config,
        )
        from isaacsim.sensors.experimental.physics import Contact, ContactSensor
        from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        from gap_isaac.config import load_task
        from gap_isaac.contact_telemetry import (
            FRANKA_CONTACT_LINKS,
            FRANKA_GROUND_MOUNT_CONTACTS,
            FRANKA_LEFT_FINGER,
            FRANKA_RIGHT_FINGER,
            ContactInterpreter,
            IsaacContactCollector,
            scale_contact_frame_count,
        )
        from gap_isaac.evaluate import (
            EpisodeStatus,
            PrivilegedEpisodeEvaluator,
            target_fits_basket_opening,
        )
        from gap_isaac.finger_drives import (
            plan_finger_preload,
            read_finger_drives,
            select_finger_actuators,
        )
        from gap_isaac.graph_policy import GraphPolicy, GraphPolicyExecutor, NodeResult
        from gap_isaac.grasp_alignment import align_open_gripper
        from gap_isaac.grocery_scene import GEOMETRY, ROBOT_BASE_POSITION_M, validate_ground_scene
        from gap_isaac.isaac_episode import IsaacEpisodeRuntime
        from gap_isaac.local_gripper import GripperSafetyPolicy, LocalGripperExecutor
        from gap_isaac.local_motion import LocalTrajectoryExecutor, MotionSafetyPolicy
        from gap_isaac.pick_place_plan import build_oracle_pick_place_plan
        from gap_isaac.perceived_feasibility import grasp_slip_check_active
        from gap_isaac.protocol import ExecuteTrajectoryRequest, TrajectoryPoint
        from gap_isaac.reset import build_episode_reset_plan

        if not open_stage(str(args.stage.resolve())):
            raise RuntimeError(f"Isaac could not open USD stage: {args.stage}")
        validate_ground_scene(omni.usd.get_context().get_stage())
        task = load_task(args.task)
        reset = build_episode_reset_plan(task, seed=args.seed, target_id=args.target)
        if (generation_receipt is None
                and graph.metadata.get("geometry_source") != "oracle-smoke-only"):
            raise RuntimeError("first graph smoke must declare oracle-smoke-only geometry")
        if perceived:
            from dataclasses import replace
            graph = replace(graph, metadata={**graph.metadata,
                "geometry_source": "saved-dino-sam3-rgbd-model-assisted",
                "graph_authoring": graph_authoring})

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / args.physics_hz,
            rendering_dt=1.0 / 60.0,
        )
        robot = SingleArticulation("/World/Franka", name="graph_franka")
        world.scene.add(robot)
        prims = {}
        for state in reset.objects:
            path = f"/World/Task/{state.object_id}"
            if state.object_id == reset.container_id:
                prim = SingleXFormPrim(path, name=f"graph_{state.object_id}")
            else:
                prim = SingleRigidPrim(path, name=f"graph_{state.object_id}")
            prims[state.object_id] = prim
            world.scene.add(prim)

        stage = omni.usd.get_context().get_stage()
        if not args.inspect_fingers:
            physics_path = world.get_physics_context().prim_path
            physics_api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(physics_path))
            contact_last = physics_api.GetSolveArticulationContactLastAttr()
            before = bool(contact_last.Get())
            # The contact-last experiment reduced measured support and failed
            # earlier. Restore the original order; test timestep independently.
            if not physics_api.CreateSolveArticulationContactLastAttr().Set(False):
                raise RuntimeError("could not restore the original contact solver order")
            diagnostics["contact_solver"] = {
                "physics_scene": physics_path,
                "solve_articulation_contact_last_before": before,
                "solve_articulation_contact_last_after": bool(contact_last.Get()),
                "physics_dt_s": float(world.get_physics_dt()),
                "requested_physics_hz": args.physics_hz,
                "solver_type": world.get_physics_context().get_solver_type(),
            }
            print(f"Contact solver: {json.dumps(diagnostics['contact_solver'])}", flush=True)
        missing_links = [
            path for path in FRANKA_CONTACT_LINKS if not stage.GetPrimAtPath(path).IsValid()
        ]
        if missing_links:
            raise RuntimeError(f"Franka stage is missing contact links: {missing_links}")
        sensors = {
            link_path: ContactSensor(
                Contact.create(
                    f"{link_path}/gap_graph_contact_sensor",
                    min_threshold=0.0,
                    max_threshold=10_000_000.0,
                )
            )
            for link_path in FRANKA_CONTACT_LINKS
        }
        interpreter = ContactInterpreter(
            stationary_mount_contacts=FRANKA_GROUND_MOUNT_CONTACTS,
            object_roots={
                state.object_id: f"/World/Task/{state.object_id}" for state in reset.objects
            },
            container_id=reset.container_id,
            robot_root="/World/Franka",
            left_finger_root=FRANKA_LEFT_FINGER,
            right_finger_root=FRANKA_RIGHT_FINGER,
            grasp_contact_frames=scale_contact_frame_count(
                task.evaluation.grasp_contact_frames, args.physics_hz
            ),
            release_contact_frames=scale_contact_frame_count(
                task.evaluation.release_contact_frames, args.physics_hz
            ),
        )
        contacts = IsaacContactCollector(
            sensors,
            lambda body_id: str(PhysicsSchemaTools.intToSdfPath(body_id)),
            interpreter,
        )

        def set_contact_offsets(root_path: str) -> None:
            pending = [stage.GetPrimAtPath(root_path)]
            while pending:
                prim = pending.pop()
                pending.extend(prim.GetChildren())
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    collision.CreateContactOffsetAttr(0.001)
                    collision.CreateRestOffsetAttr(0.0)

        for contact_root in (
            FRANKA_LEFT_FINGER,
            FRANKA_RIGHT_FINGER,
            f"/World/Task/{reset.target_id}",
        ):
            set_contact_offsets(contact_root)

        stage.DefinePrim("/World/PhysicsMaterials", "Scope")
        grip_material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/GapGripperPad")
        grip_material_api = UsdPhysics.MaterialAPI.Apply(grip_material.GetPrim())
        grip_material_api.CreateStaticFrictionAttr().Set(1.2)
        grip_material_api.CreateDynamicFrictionAttr().Set(1.0)
        grip_material_api.CreateRestitutionAttr().Set(0.0)
        for grip_root in (
            FRANKA_LEFT_FINGER,
            FRANKA_RIGHT_FINGER,
            f"/World/Task/{reset.target_id}",
        ):
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(grip_root)).Bind(
                grip_material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )

        # Audit the actual colliders, including instance proxies. A binding on
        # a parent alone is not evidence of the resolved physics material.
        material_bindings = []
        diagnostics["grip_collision_materials"] = material_bindings
        for root in (FRANKA_LEFT_FINGER, FRANKA_RIGHT_FINGER, f"/World/Task/{reset.target_id}"):
            root_colliders = 0
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                root_colliders += 1
                material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
                    materialPurpose="physics"
                )
                path = str(material.GetPath()) if material else None
                material_bindings.append(
                    {"collider": str(prim.GetPath()), "physics_material": path}
                )
                if path != str(grip_material.GetPath()):
                    raise RuntimeError(f"grip collider {prim.GetPath()} resolved material {path}")
            if root_colliders == 0:
                raise RuntimeError(f"no grip colliders found under {root}")
        diagnostics["grip_material"] = {"static_friction": 1.2, "dynamic_friction": 1.0}

        world.reset()
        diagnostics["target_mass_kg"] = float(prims[reset.target_id].get_mass())
        robot_base_position, robot_base_orientation = robot.get_world_pose()
        live_base = tuple(float(value) for value in robot_base_position)
        print(f"Live Franka base position: {live_base}", flush=True)
        if any(
            abs(live_base[index] - expected) > 1e-3
            for index, expected in enumerate(ROBOT_BASE_POSITION_M)
        ):
            raise RuntimeError(f"live Franka base is {live_base}, expected {ROBOT_BASE_POSITION_M}")

        controller = robot.get_articulation_controller()
        position_iterations = int(robot.get_solver_position_iteration_count())
        velocity_iterations = int(robot.get_solver_velocity_iteration_count())
        diagnostics["articulation_solver_iterations"] = {
            "position_before": position_iterations,
            "velocity_before": velocity_iterations,
            "position_after": int(robot.get_solver_position_iteration_count()),
            "velocity_after": int(robot.get_solver_velocity_iteration_count()),
        }
        diagnostics["arm_joint_usd_audit"] = [
            {"path": str(prim.GetPath()), "schemas": list(prim.GetAppliedSchemas()),
             "attributes": {attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()
                            if any(key in attr.GetName().lower() for key in
                                   ("drive", "mimic", "friction", "limit"))}}
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith("/World/Franka/")
            and prim.GetName() in tuple(robot.dof_names[:7])
        ]
        finger_drives = read_finger_drives(stage, robot, controller)
        diagnostics["finger_drives"] = [drive.as_dict() for drive in finger_drives]
        print(f"Finger drive inspection: {json.dumps(diagnostics['finger_drives'])}", flush=True)
        active_fingers = select_finger_actuators(finger_drives)
        diagnostics["active_finger_names"] = [drive.name for drive in active_fingers]
        if args.inspect_fingers:
            output = {"ok": True, "test": "finger-drive-inspection", "motion_enabled": False}
            write_result(output)
            print(json.dumps(output, indent=2, sort_keys=True), flush=True)
            return 0

        episode_runtime = IsaacEpisodeRuntime(robot, prims, reset)
        episode_runtime.apply_reset()

        # Joint 5 retained a 28 mrad error at the loaded lift waypoint despite
        # two seconds of settling. Increase this drive's stiffness without
        # changing its effort limit or the finger drives. Scale damping with
        # sqrt(stiffness) to retain the nominal damping ratio.
        original_kps, original_kds = controller.get_gains()
        kps = np.asarray(original_kps, dtype=np.float64).copy()
        kds = np.asarray(original_kds, dtype=np.float64).copy()
        # Bounded position tracking with the asset's very overdamped drives
        # failed even during stationary pauses. Tune damping only; retain
        # stiffness, effort limits, and all tracking/collision checks.
        kds[:7] *= .05
        diagnostics["arm_damping_tuning"] = {
            "scale": .05,
            "before": [float(v) for v in original_kds[:7]],
            "after_before_wrist_scaling": [float(v) for v in kds[:7]],
            "stiffness_changed_by_this_tuning": False,
            "effort_limits_changed": False,
        }
        wrist_index = tuple(robot.dof_names).index("panda_joint5")
        if (
            not math.isfinite(float(kps[wrist_index]))
            or not math.isfinite(float(kds[wrist_index]))
            or kps[wrist_index] <= 0
            or kds[wrist_index] < 0
        ):
            raise RuntimeError("joint 5 must expose finite position drive gains")
        wrist_drive = {
            "joint": "panda_joint5",
            "stiffness_before": float(kps[wrist_index]),
            "damping_before": float(kds[wrist_index]),
            "stiffness_scale": 2.0,
            "damping_scale": math.sqrt(2.0),
        }
        kps[wrist_index] *= wrist_drive["stiffness_scale"]
        kds[wrist_index] *= wrist_drive["damping_scale"]
        force_multiplier = 2.0 if len(active_fingers) == 1 else 1.0
        nominal_finger_force_n = 3.5
        # Restore the inspected 400 N/m leader instead of sizing a 14,000
        # N/m spring around an arbitrary 0.5 mm target offset. At 400 N/m,
        # 7 N actuator preload requires 17.5 mm *command* offset. Contact
        # blocks actual finger travel; the effort limit remains 7.2 N.
        deflection_cap_m = 0.0175
        for drive in plan_finger_preload(
            finger_drives,
            force_per_finger_n=nominal_finger_force_n,
            maximum_deflection_m=deflection_cap_m,
        ):
            # Keep the original stiffness when it can supply the nominal
            # preload within the bounded command offset. No effort increase.
            kps[drive.index] = drive.stiffness_n_m
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        applied_kps, applied_kds = controller.get_gains()
        maximum_efforts = controller.get_max_efforts()
        diagnostics["arm_drive_audit"] = {
            "names": [str(name) for name in robot.dof_names[:7]],
            "stiffness": [float(v) for v in applied_kps[:7]],
            "damping": [float(v) for v in applied_kds[:7]],
            "effort_limits_nm": [float(v) for v in maximum_efforts[:7]],
            "self_collisions_enabled": bool(robot.get_enabled_self_collisions()),
        }
        wrist_drive.update(
            stiffness_after=float(applied_kps[wrist_index]),
            damping_after=float(applied_kds[wrist_index]),
            effort_limit_nm=(
                float(maximum_efforts[wrist_index]) if maximum_efforts is not None else None
            ),
        )
        print(f"Graph wrist drive: {json.dumps(wrist_drive, sort_keys=True)}", flush=True)
        diagnostics["applied_finger_drives"] = [
            drive.as_dict() for drive in read_finger_drives(stage, robot, controller)
        ]
        print(
            f"Applied finger drives: {json.dumps(diagnostics['applied_finger_drives'])}", flush=True
        )

        contacts.reset()
        physics_dt_s = float(world.get_physics_dt())
        for _ in range(episode_runtime.settle_step_count(physics_dt_s)):
            world.step(render=False)

        evaluator = PrivilegedEpisodeEvaluator(task, reset)
        elapsed_s = reset.settle_time_s
        latest_telemetry = contacts.capture()
        last_capture_time_s = elapsed_s
        verify_hold_during_motion = False
        grasp_relative_reference = None
        maximum_grasp_slip_m = 0.0
        contact_history = []
        diagnostics["recent_contact_frames"] = contact_history
        latest_evaluation = evaluator.update(
            episode_runtime.capture_frame(
                elapsed_s,
                held_object_id=latest_telemetry.held_object_id,
                collisions=latest_telemetry.collisions,
            )
        )
        if latest_evaluation.terminal:
            raise RuntimeError(
                f"episode was terminal before graph execution: {latest_evaluation.as_dict()}"
            )

        def step() -> None:
            nonlocal elapsed_s
            world.step(render=False)
            elapsed_s += physics_dt_s

        def capture():
            nonlocal latest_telemetry, latest_evaluation, last_capture_time_s
            # Debounce once per physics frame, even if multiple checks read it.
            if last_capture_time_s == elapsed_s:
                return latest_telemetry
            last_capture_time_s = elapsed_s
            latest_telemetry = contacts.capture()
            latest_evaluation = evaluator.update(
                episode_runtime.capture_frame(
                    elapsed_s,
                    held_object_id=latest_telemetry.held_object_id,
                    collisions=latest_telemetry.collisions,
                )
            )
            contact_history.append(
                {
                    "elapsed_s": elapsed_s,
                    "held_object_id": latest_telemetry.held_object_id,
                    "contacts": [
                        {
                            "body_a": event.body_a,
                            "body_b": event.body_b,
                            "impulse_ns": event.impulse_ns,
                            "allowed": event.allowed,
                        }
                        for event in latest_telemetry.collisions
                    ],
                }
            )
            # Bound diagnostics to the last half-second of physical contact.
            del contact_history[: -max(1, math.ceil(0.5 / physics_dt_s))]
            return latest_telemetry

        alignment_sampler = None
        descent_guard = None
        descent_trace_state = {}

        def safety_check() -> str | None:
            nonlocal maximum_grasp_slip_m
            capture()
            if alignment_sampler is not None:
                alignment_sampler()
            if latest_evaluation.status is EpisodeStatus.FAILED:
                return (
                    f"episode failed: {latest_evaluation.termination_reason}: "
                    f"{latest_evaluation.detail}"
                )
            if descent_guard is not None:
                from gap_isaac.cartesian_descent import down_orientation_error
                ee_position, ee_rotation = ik.compute_end_effector_pose()
                xy_error = float(np.linalg.norm(np.asarray(ee_position)[:2] - descent_guard[:2]))
                orientation_error = down_orientation_error(ee_rotation)
                applied = controller.get_applied_action()
                measured = robot.get_joint_positions()
                velocities = robot.get_joint_velocities()
                arm_names = tuple(str(name) for name in lula.get_joint_names())
                arm_indices = tuple(tuple(robot.dof_names).index(name) for name in arm_names)
                commanded = None if applied is None else applied.joint_positions
                diagnostics["cartesian_descent"]["latest_tracking"] = {
                    "elapsed_s": elapsed_s,
                    "xy_error_m": xy_error,
                    "orientation_error_rad": orientation_error,
                    "joint_names": arm_names,
                    "measured_positions_rad": [float(measured[i]) for i in arm_indices],
                    "commanded_positions_rad": (
                        [float(commanded[i]) for i in arm_indices]
                        if commanded is not None else None),
                    "commanded_velocities_rad_s": (
                        [float(applied.joint_velocities[i]) for i in arm_indices]
                        if applied is not None and applied.joint_velocities is not None else None),
                    "joint_velocities_rad_s": [
                        float(velocities[i]) for i in arm_indices],
                    "measured_efforts_nm": [
                        float(v) for v in robot.get_measured_joint_efforts(
                            joint_indices=np.asarray(arm_indices, dtype=np.int64))],
                }
                target_position, target_rotation = prims[reset.target_id].get_world_pose()
                trace_sample = {
                    **diagnostics["cartesian_descent"]["latest_tracking"],
                    **descent_trace_state,
                    "tcp_position_m": [float(v) for v in ee_position],
                    "tcp_rotation_world": np.asarray(ee_rotation).tolist(),
                    "finger_positions_m": [float(measured[d.index]) for d in finger_drives],
                    "finger_command_positions_m": (
                        [float(commanded[d.index]) for d in finger_drives]
                        if commanded is not None else None),
                    "contacts": contact_history[-1]["contacts"] if contact_history else [],
                    "held_object_id": latest_telemetry.held_object_id,
                    "diagnostic_only_target_position_m": [float(v) for v in target_position],
                    "diagnostic_only_target_orientation_wxyz": [float(v) for v in target_rotation],
                }
                descent_trace_stream.write(json.dumps(trace_sample) + "\n")
                if xy_error > .008 or orientation_error > .05:
                    return (f"Cartesian descent tracking guard: xy_error={xy_error:.6f} m; "
                            f"orientation_error={orientation_error:.6f} rad")
            if verify_hold_during_motion and latest_telemetry.held_object_id != reset.target_id:
                return hold_failure_detail("motion")
            if grasp_slip_check_active(args.grasp_validation, perceived, verify_hold_during_motion,
                                       grasp_relative_reference is not None):
                position, _ = prims[reset.target_id].get_world_pose()
                ee_position, _ = ik.compute_end_effector_pose(position_only=True)
                slip_m = float(
                    np.linalg.norm(
                        np.asarray(position) - np.asarray(ee_position) - grasp_relative_reference
                    )
                )
                maximum_grasp_slip_m = max(maximum_grasp_slip_m, slip_m)
                if slip_m > 0.015:
                    return f"target slipped {slip_m:.6f} m relative to gripper during motion"
            return None

        def action_factory(targets, indices):
            return ArticulationAction(
                joint_positions=np.asarray(targets, dtype=np.float64),
                joint_indices=np.asarray(indices, dtype=np.int64),
            )

        motion_policy = MotionSafetyPolicy(
            max_trajectory_duration_s=12.0,
            max_command_speed_rad_s=0.35,
            robot_velocity_scale=0.25,
            final_position_tolerance=0.02,
            final_settle_time_s=0.5,
            max_final_settle_time_s=4.0,
        )
        trajectory = LocalTrajectoryExecutor(
            robot,
            physics_dt_s=physics_dt_s,
            step=step,
            action_factory=action_factory,
            safety_check=safety_check,
            policy=motion_policy,
            allowed_joint_names=reset.robot_joint_names[:7],
        )
        gripper_policy = GripperSafetyPolicy(
            grasp_force_n=nominal_finger_force_n,
            max_grasp_deflection_m=deflection_cap_m,
            grasp_settle_s=0.5,
            # A single mimic leader works against both finger contact forces.
            grasp_force_multiplier=force_multiplier,
        )
        finger_stiffness = tuple(float(applied_kps[drive.index]) for drive in active_fingers)
        gripper = LocalGripperExecutor(
            robot,
            physics_dt_s=physics_dt_s,
            step=step,
            action_factory=action_factory,
            contact_capture=capture,
            policy=gripper_policy,
            finger_stiffness_n_m=finger_stiffness,
            controlled_finger_names=tuple(drive.name for drive in active_fingers),
        )

        kinematics_config = load_supported_lula_kinematics_solver_config("Franka")
        lula = LulaKinematicsSolver(**kinematics_config)
        lula.set_robot_base_pose(robot_base_position, robot_base_orientation)
        ik = ArticulationKinematicsSolver(robot, lula, "right_gripper")
        down_orientation = np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float64)
        settled_positions = {}
        for object_id, prim in prims.items():
            position, _ = prim.get_world_pose()
            settled_positions[object_id] = tuple(float(value) for value in position)
        if perceived:
            if args.baseline_diagnostics:
                from gap_isaac.perceived_feasibility import build_perceived_grasp_plan, validate_grasp_gate
            else:
                from gap_isaac.perceived_feasibility_v24 import build_perceived_grasp_plan, validate_grasp_gate
            from gap_isaac.observation_bundle import ObservationBundle
            from isaacsim.sensors.camera import Camera
            target_geometry = json.loads(args.target_geometry.read_text())
            basket_geometry = json.loads(args.basket_geometry.read_text())
            plan, geometry_checks = build_perceived_grasp_plan(
                target_geometry, basket_geometry, task.evaluation.basket_wall_thickness_m,
                task.evaluation.containment_margin_m)
            if not args.grasp_validation:
                validate_grasp_gate(json.loads(args.grasp_result.read_text()), geometry_checks["waypoints_m"])
                diagnostics["physical_grasp_gate"] = str(args.grasp_result.resolve())
            bundle = ObservationBundle.load(Path(target_geometry["bundle"]))
            import hashlib
            if hashlib.sha256((Path(target_geometry["bundle"]) / "rgb.png").read_bytes()).hexdigest() != target_geometry["image_sha256"]:
                raise RuntimeError("Perceived geometry does not match its verified saved capture")
            camera = Camera("/World/Camera", resolution=(640, 480))
            camera_position, camera_orientation = camera.get_world_pose(camera_axes="ros")
            saved_pose = bundle.observation.camera_pose
            q = saved_pose.orientation
            camera_position_error = float(np.linalg.norm(camera_position - np.array(
                [saved_pose.position.x, saved_pose.position.y, saved_pose.position.z])))
            camera_angle = float(2 * np.arccos(np.clip(abs(np.dot(camera_orientation,
                np.array([q.w, q.x, q.y, q.z]))), 0, 1)))
            usd_camera = UsdGeom.Camera.Get(stage, "/World/Camera")
            focal_error = abs(640 * usd_camera.GetFocalLengthAttr().Get() /
                              usd_camera.GetHorizontalApertureAttr().Get() - bundle.observation.camera.fx)
            target_error = float(np.linalg.norm(np.asarray(plan.target_center.as_tuple()) -
                                                settled_positions[reset.target_id]))
            bc = basket_geometry["obb"]["center"]
            basket_error = float(np.linalg.norm(np.array([bc["x"], bc["y"]]) -
                                                np.asarray(settled_positions[reset.container_id])[:2]))
            diagnostics["perceived_geometry"] = {
                "geometry_source": "saved-dino-sam3-rgbd-model-assisted",
                "target_file": str(args.target_geometry.resolve()),
                "basket_file": str(args.basket_geometry.resolve()),
                "target_center_error_m": target_error, "basket_center_xy_error_m": basket_error,
                "camera_position_error_m": camera_position_error,
                "camera_orientation_error_rad": camera_angle, "camera_focal_error_px": float(focal_error),
                "pose_corrections_from_oracle": False, "checks": geometry_checks}
            if (target_error > .02 or basket_error > .025 or camera_position_error > .001
                    or camera_angle > .001 or focal_error > 1):
                raise RuntimeError("Cached perceived geometry no longer aligns with the live settled scene")
        else:
            plan = build_oracle_pick_place_plan(task, reset, settled_positions)
        diagnostics["placement_plan"] = {
            "strategy": "release-above-rim",
            "basket_rim_z_m": plan.basket_rim_z_m,
            "planned_target_bottom_clearance_m": plan.release_bottom_clearance_m,
            "release_tcp_position_m": list(plan.place.as_tuple()),
        }
        gripper_preload = {
            "nominal_force_per_finger_n": gripper_policy.grasp_force_n,
            "maximum_deflection_per_finger_m": gripper_policy.max_grasp_deflection_m,
            "deflection_is_command_offset_not_collision_penetration": True,
            "finger_stiffness_n_m": list(finger_stiffness),
            "controlled_finger_names": [drive.name for drive in active_fingers],
            "actuator_force_multiplier": gripper_policy.grasp_force_multiplier,
            "grasp_frame_above_object_center_m": plan.grasp_frame_above_object_center_m,
        }
        print(f"Graph gripper settings: {json.dumps(gripper_preload, sort_keys=True)}", flush=True)

        def move_to(
            command_id: str, position: tuple[float, float, float], *, precise: bool = False
        ) -> NodeResult:
            nonlocal alignment_sampler
            ik_action, solved = ik.compute_inverse_kinematics(
                target_position=np.asarray(position, dtype=np.float64),
                target_orientation=down_orientation,
                position_tolerance=0.0001 if precise else 0.005,
                orientation_tolerance=0.002 if precise else 0.05,
            )
            if not solved:
                raise RuntimeError(f"IK did not converge for {command_id} at {position}")
            if ik_action.joint_positions is None:
                raise RuntimeError(f"IK returned no joint target for {command_id}")
            if ik_action.joint_indices is None:
                solver_names = tuple(str(name) for name in lula.get_joint_names())
                indices = tuple(tuple(robot.dof_names).index(name) for name in solver_names)
            else:
                indices = tuple(int(index) for index in ik_action.joint_indices)
            targets = tuple(float(value) for value in ik_action.joint_positions)
            names = tuple(str(robot.dof_names[index]) for index in indices)
            current = robot.get_joint_positions()
            initial_targets = None
            if verify_hold_during_motion:
                applied_action = controller.get_applied_action()
                if applied_action is None or applied_action.joint_positions is None:
                    raise RuntimeError("loaded motion requires the existing arm holding targets")
                initial_targets = tuple(
                    float(applied_action.joint_positions[index]) for index in indices
                )
                diagnostics.setdefault("loaded_motion_starts", []).append(
                    {
                        "command_id": command_id,
                        "elapsed_s": elapsed_s,
                        "joint_names": list(names),
                        "holding_targets_rad": list(initial_targets),
                        "measured_positions_rad": [float(current[index]) for index in indices],
                        "destination_targets_rad": list(targets),
                    }
                )
            durations = []
            for offset, (index, target) in enumerate(zip(indices, targets, strict=True)):
                maximum = min(
                    motion_policy.max_command_speed_rad_s,
                    float(robot.dof_properties[index]["maxVelocity"])
                    * motion_policy.robot_velocity_scale,
                )
                delta = abs(target - float(current[index]))
                if initial_targets is not None:
                    delta = max(delta, abs(target - initial_targets[offset]))
                durations.append(delta / maximum)
            minimum_duration_s = 2.0 if args.grasp_validation and verify_hold_during_motion else 1.0
            duration_s = max(minimum_duration_s, max(durations, default=0.0) * 1.10)
            if duration_s > motion_policy.max_trajectory_duration_s:
                raise RuntimeError(
                    f"{command_id} requires {duration_s:.3f}s, above the local motion bound"
                )
            request = ExecuteTrajectoryRequest(
                command_id=command_id,
                joint_names=names,
                points=(TrajectoryPoint(time_from_start_s=duration_s, positions_rad=targets),),
                timeout_s=duration_s + motion_policy.max_final_settle_time_s + 1.0,
            )
            active_trajectory = trajectory
            if command_id == "graph-align":
                from gap_isaac.local_motion_v25 import LocalTrajectoryExecutor as EndpointExecutor
                from gap_isaac.cartesian_descent import down_orientation_error
                from isaacsim.core.utils.rotations import quat_to_rot_matrix
                hand = SingleXFormPrim("/World/Franka/panda_hand", name="alignment_hand")
                samples = []
                diagnostics["alignment_endpoint"] = {
                    "command_id": command_id, "destination_m": list(position),
                    "final_targets_rad": list(targets), "joint_names": list(names),
                    "path_duration_s": duration_s, "minimum_settle_s": .5,
                    "maximum_settle_s": 4.0, "samples": samples,
                    "independent_tcp_definition": "panda_hand + local Z 0.1m; URDF right_gripper offset",
                }
                begun = elapsed_s
                def sample_alignment():
                    actual = robot.get_joint_positions()
                    velocity = robot.get_joint_velocities()
                    applied = controller.get_applied_action()
                    measured = [float(actual[i]) for i in indices]
                    commands = [float(applied.joint_positions[i]) for i in indices]
                    commanded_fk, _ = lula.compute_forward_kinematics("right_gripper", np.asarray(commands))
                    measured_fk, _ = lula.compute_forward_kinematics("right_gripper", np.asarray(measured))
                    final_fk, _ = lula.compute_forward_kinematics("right_gripper", np.asarray(targets))
                    hp, hq = hand.get_world_pose()
                    independent_tcp = np.asarray(hp) + quat_to_rot_matrix(hq) @ np.asarray([0., 0., .1])
                    tcp_velocity = (None if not samples else
                        ((np.asarray(measured_fk) - np.asarray(samples[-1]["measured_fk_m"])) /
                         (elapsed_s - samples[-1]["elapsed_s"])).tolist())
                    sample = {"elapsed_s": elapsed_s, "align_elapsed_s": elapsed_s - begun,
                        "settle_elapsed_s": max(0., elapsed_s - begun - duration_s),
                        "measured_joints_rad": measured, "commanded_joints_rad": commands,
                        "measured_joint_velocities_rad_s": [float(velocity[i]) for i in indices],
                        "commanded_fk_m": np.asarray(commanded_fk).tolist(),
                        "final_command_fk_m": np.asarray(final_fk).tolist(),
                        "measured_fk_m": np.asarray(measured_fk).tolist(),
                        "independent_tcp_m": independent_tcp.tolist(),
                        "tcp_velocity_m_s": tcp_velocity,
                        "xy_error_m": float(np.linalg.norm(np.asarray(measured_fk)[:2] - np.asarray(position)[:2])),
                        "final_command_fk_error_m": float(np.linalg.norm(np.asarray(final_fk) - np.asarray(position))),
                        "model_tcp_difference_m": float(np.linalg.norm(np.asarray(measured_fk) - independent_tcp)),
                        "max_joint_error_rad": max(abs(a-b) for a,b in zip(measured, targets)),
                        "contacts": contact_history[-1]["contacts"] if contact_history else []}
                    samples.append(sample)
                alignment_sampler = sample_alignment
                def endpoint_ready():
                    pose, rotation = ik.compute_end_effector_pose()
                    xy = float(np.linalg.norm(np.asarray(pose)[:2] - np.asarray(position)[:2]))
                    angle = down_orientation_error(rotation)
                    # Reuse unchanged descent entry AND progress readiness conditions.
                    return xy <= .005 and xy < .003 and angle < .01
                active_trajectory = EndpointExecutor(
                    robot, physics_dt_s=physics_dt_s, step=step, action_factory=action_factory,
                    safety_check=safety_check, policy=motion_policy,
                    final_completion_check=endpoint_ready,
                    allowed_joint_names=reset.robot_joint_names[:7])
            try:
                result = active_trajectory.execute(request, initial_targets=initial_targets)
            finally:
                alignment_sampler = None
            if result.status != "succeeded":
                actual_position, _ = ik.compute_end_effector_pose(position_only=True)
                target_position, _ = prims[reset.target_id].get_world_pose()
                raise RuntimeError(
                    f"{command_id} {result.status}: {result.detail}; "
                    f"end_effector_position={tuple(float(v) for v in actual_position)}, "
                    f"target_position={tuple(float(v) for v in target_position)}, "
                    f"held_object={latest_telemetry.held_object_id or 'none'}, "
                    f"gripper_separation={gripper.separation_m():.6f} m, "
                    f"joint5_velocity_rad_s="
                    f"{float(robot.get_joint_velocities()[wrist_index]):.6f}"
                )
            actual_position, _ = ik.compute_end_effector_pose(position_only=True)
            position_error_m = float(
                np.linalg.norm(actual_position - np.asarray(position, dtype=np.float64))
            )
            if position_error_m > 0.025:
                raise RuntimeError(
                    f"{command_id} position error {position_error_m:.6f} m exceeds tolerance"
                )
            return NodeResult(
                detail=(
                    f"reached {position} with {position_error_m:.6f} m position error; "
                    f"{result.detail}"
                )
            )

        def hold_failure_detail(phase: str) -> str:
            target_position, _ = prims[reset.target_id].get_world_pose()
            target_velocity = prims[reset.target_id].get_linear_velocity()
            return (
                f"{phase} lost target hold: {latest_telemetry.held_object_id or 'none'}; "
                f"target_position={tuple(float(v) for v in target_position)}, "
                f"target_velocity={tuple(float(v) for v in target_velocity)}, "
                f"gripper_separation={gripper.separation_m():.6f} m, "
                f"raw_contacts={latest_telemetry.raw_contact_count}"
            )

        def require_hold(phase: str) -> None:
            capture()
            if latest_telemetry.held_object_id != reset.target_id:
                raise RuntimeError(hold_failure_detail(phase))

        def open_gripper(_state) -> NodeResult:
            result = gripper.move("graph-open", open_gripper=True)
            if result.status != "succeeded":
                raise RuntimeError(f"open gripper {result.status}: {result.detail}")
            return NodeResult(detail="gripper opened")

        def align_over_target(_state) -> NodeResult:
            return move_to("graph-align", plan.align.as_tuple())

        def descend_to_grasp(_state) -> NodeResult:
            nonlocal descent_guard, descent_trace_stream
            from gap_isaac.cartesian_descent import plan_descent

            names = tuple(str(name) for name in lula.get_joint_names())
            indices = tuple(tuple(robot.dof_names).index(name) for name in names)
            current = robot.get_joint_positions()
            initial = tuple(float(current[index]) for index in indices)
            start, _ = ik.compute_end_effector_pose(position_only=True)
            destination = plan.grasp.as_tuple()
            diagnostics["descent_start"] = {"position_m": list(start),
                                            "destination_m": list(destination),
                                            "measured_joints_rad": list(initial)}

            def solve(position, warm):
                return lula.compute_inverse_kinematics(
                    "right_gripper", np.asarray(position, dtype=np.float64), down_orientation,
                    warm_start=np.asarray(warm, dtype=np.float64),
                    position_tolerance=.0001, orientation_tolerance=.002,
                )

            def forward(joints):
                return lula.compute_forward_kinematics(
                    "right_gripper", np.asarray(joints, dtype=np.float64)
                )

            descent = plan_descent(
                start=tuple(float(v) for v in start), destination=destination,
                initial_joints=initial, joint_names=names,
                joint_speed_limits=tuple(min(
                    motion_policy.max_command_speed_rad_s,
                    float(robot.dof_properties[index]["maxVelocity"])
                    * motion_policy.robot_velocity_scale,
                ) for index in indices),
                solve=solve, forward=forward,
                max_duration_s=motion_policy.max_trajectory_duration_s,
                settle_timeout_s=motion_policy.max_final_settle_time_s,
                cartesian_speed_m_s=.035,
            )
            diagnostics["cartesian_descent"] = {
                "start_m": list(start), "destination_m": list(destination),
                "waypoints_m": descent.positions_m,
                "joint_points": [point.model_dump() for point in descent.request.points],
                "duration_s": descent.request.points[-1].time_from_start_s,
                "cartesian_speed_m_s": .035,
                "maximum_fk_error_m": descent.maximum_fk_error_m,
                "maximum_orientation_error_rad": descent.maximum_orientation_error_rad,
                "warm_start": "previous solution; branch jumps rejected",
                "xy_tracking_limit_m": .008, "orientation_tracking_limit_rad": .05,
                "pose_source": "fixed perceived geometry" if perceived else "oracle-smoke-only",
            }
            print(f"Executing Cartesian descent: {len(descent.positions_m)} waypoints; "
                  f"{descent.request.points[-1].time_from_start_s:.3f} s.", flush=True)
            trace_path = args.result_file.with_suffix(".descent.jsonl")
            descent_trace_stream = trace_path.open("x", encoding="utf-8", buffering=1)
            diagnostics["descent_trace_file"] = str(trace_path)
            descent_trace_state.update(path_time_s=0.0, descent_elapsed_s=0.0,
                                       progress_allowed=None, pause_reasons=[])
            descent_guard = np.asarray(destination, dtype=np.float64)
            try:
                def descent_action(targets, velocities, joint_indices):
                    return ArticulationAction(
                        joint_positions=np.asarray(targets, dtype=np.float64),
                        joint_velocities=np.asarray(velocities, dtype=np.float64),
                        joint_indices=np.asarray(joint_indices, dtype=np.int64),
                    )

                def descent_progress_allowed():
                    from gap_isaac.cartesian_descent import down_orientation_error
                    position, rotation = ik.compute_end_effector_pose()
                    lateral_error = float(np.linalg.norm(
                        np.asarray(position)[:2] - descent_guard[:2]))
                    angle = down_orientation_error(rotation)
                    allowed = lateral_error < .003 and angle < .01
                    descent_trace_state.update(
                        progress_allowed=allowed,
                        pre_step_xy_error_m=lateral_error,
                        pre_step_orientation_error_rad=angle,
                        pause_reasons=(["xy"] if lateral_error >= .003 else [])
                                      + (["orientation"] if angle >= .01 else []),
                    )
                    if allowed:
                        descent_trace_state["path_time_s"] = min(
                            descent_trace_state["path_time_s"] + physics_dt_s,
                            descent.request.points[-1].time_from_start_s)
                    return allowed

                def descent_step():
                    step()
                    descent_trace_state["descent_elapsed_s"] += physics_dt_s
                    descent_trace_state["phase"] = (
                        "settle" if descent_trace_state["path_time_s"] >=
                        descent.request.points[-1].time_from_start_s else "path")

                descent_executor = LocalTrajectoryExecutor(
                    robot, physics_dt_s=physics_dt_s, step=descent_step,
                    action_factory=action_factory,
                    velocity_action_factory=descent_action,
                    progress_check=descent_progress_allowed,
                    safety_check=safety_check, policy=motion_policy,
                    allowed_joint_names=reset.robot_joint_names[:7],
                )
                diagnostics["cartesian_descent"]["velocity_feed_forward"] = True
                diagnostics["cartesian_descent"]["tracking_paced"] = True
                bounded_request = descent.request.model_copy(update={
                    "timeout_s": motion_policy.max_trajectory_duration_s
                    + motion_policy.max_final_settle_time_s + 1.0})
                result = descent_executor.execute(bounded_request)
                actual, _ = ik.compute_end_effector_pose(position_only=True)
                error = float(np.linalg.norm(np.asarray(actual) - descent_guard))
                diagnostics["cartesian_descent"].update(
                    status=result.status, detail=result.detail, final_position_error_m=error)
                if result.status != "succeeded" or error > .005:
                    raise RuntimeError(f"Cartesian descent {result.status}: {result.detail}; "
                                       f"final_position_error={error:.6f} m")
                return NodeResult(detail=f"Cartesian descent reached grasp with {error:.6f} m error")
            finally:
                descent_guard = None
                descent_trace_stream.close()
                descent_trace_stream = None

        def close_gripper(_state) -> NodeResult:
            nonlocal verify_hold_during_motion, grasp_relative_reference
            alignment_observations = []
            diagnostics["pre_close_alignment"] = alignment_observations

            def measure_alignment():
                # Let the arm settle with the fingers open; a single waypoint
                # sample missed the observed drift during the five-second close.
                for _ in range(math.ceil(1.5 / physics_dt_s)):
                    step()
                    error = safety_check()
                    if error:
                        raise RuntimeError(error)
                actual_position, _ = ik.compute_end_effector_pose(position_only=True)
                if perceived:
                    desired = plan.grasp.as_tuple()
                else:
                    target_position, _ = prims[reset.target_id].get_world_pose()
                    desired = (float(target_position[0]), float(target_position[1]),
                               float(target_position[2]) + plan.grasp_frame_above_object_center_m)
                return desired, tuple(float(value) for value in actual_position)

            align_open_gripper(
                plan.grasp.as_tuple(),
                measure=measure_alignment,
                move=lambda position: move_to("graph-grasp-center", position, precise=True),
                observations=alignment_observations,
            )
            print(
                f"Pre-close alignment error: {alignment_observations[-1]['error_m']:.6f} m "
                f"after {len(alignment_observations) - 1} corrections",
                flush=True,
            )
            result = gripper.move(
                "graph-close",
                open_gripper=False,
                require_held_object_id=reset.target_id,
            )
            diagnostics["grasp_preload_transition"] = dict(gripper.preload_diagnostics)
            if result.status != "succeeded":
                raise RuntimeError(f"close gripper {result.status}: {result.detail}")
            verify_hold_during_motion = True
            if perceived:
                object_position, _ = prims[reset.target_id].get_world_pose()
                tcp_position, _ = ik.compute_end_effector_pose(position_only=True)
                grasp_relative_reference = np.asarray(object_position) - np.asarray(tcp_position)
            return NodeResult(
                detail=result.detail,
                outputs={"grasp_separation_m": gripper.separation_m()},
            )

        def lift_grasped(_state) -> NodeResult:
            result = move_to("graph-lift", plan.lift.as_tuple())
            require_hold("lift")
            return result

        def move_over_basket(_state) -> NodeResult:
            result = move_to("graph-transport", plan.transport.as_tuple())
            require_hold("transport")
            return result

        def position_for_release(_state) -> NodeResult:
            result = move_to("graph-position-for-release", plan.place.as_tuple())
            require_hold("position for release")
            return result

        def release_object(_state) -> NodeResult:
            nonlocal verify_hold_during_motion
            require_hold("before release")
            frame = episode_runtime.capture_frame(elapsed_s)
            objects = {state.object_id: state for state in frame.objects}
            target = objects[reset.target_id]
            container = objects[reset.container_id]
            bottom_clearance = (
                target.position_m[2] - GEOMETRY[reset.target_id].size_m[2] / 2 - plan.basket_rim_z_m
            )
            fits_opening = target_fits_basket_opening(target, container, task.evaluation)
            diagnostics["pre_release_geometry"] = {
                "target_bottom_clearance_m": bottom_clearance,
                "target_footprint_fits_opening": fits_opening,
            }
            # do not open the gripper unless the object clears the basket rim
            if bottom_clearance < 0.01 or not fits_opening:
                raise RuntimeError(
                    f"unsafe release geometry: bottom clearance={bottom_clearance:.6f} m; "
                    f"footprint fits basket opening={fits_opening}"
                )
            verify_hold_during_motion = False
            result = gripper.move("graph-release", open_gripper=True)
            if result.status != "succeeded":
                raise RuntimeError(f"release gripper {result.status}: {result.detail}")
            return NodeResult(detail="gripper opened for release")

        def retract_from_basket(_state) -> NodeResult:
            return move_to("graph-retract", plan.retract.as_tuple())

        def settle_release(_state) -> NodeResult:
            # only the evaluator can confirm a completed placement
            settle_steps = math.ceil(2.0 / physics_dt_s)
            for _ in range(settle_steps):
                step()
                capture()
                if latest_evaluation.status is EpisodeStatus.FAILED:
                    raise RuntimeError(
                        f"release evaluation failed: {latest_evaluation.termination_reason}: "
                        f"{latest_evaluation.detail}"
                    )
                if latest_evaluation.status is EpisodeStatus.SUCCEEDED:
                    return NodeResult(detail="privileged evaluator verified containment")
            raise RuntimeError(
                f"release did not satisfy evaluator checkpoints: {latest_evaluation.checkpoints}"
            )

        if args.grasp_validation:
            checkpoints = []
            diagnostics["physical_grasp_checkpoints"] = checkpoints
            diagnostics["gripper_preload"] = gripper_preload
            initial_target_position = np.asarray(settled_positions[reset.target_id])

            def checkpoint(phase: str, *, validate: bool = True) -> None:
                nonlocal grasp_relative_reference
                target_position, target_orientation = prims[reset.target_id].get_world_pose()
                ee_position, ee_rotation = ik.compute_end_effector_pose()
                relative = np.asarray(target_position) - np.asarray(ee_position)
                if phase == "close" and validate:
                    grasp_relative_reference = relative.copy()
                slip_m = (
                    float(np.linalg.norm(relative - grasp_relative_reference))
                    if grasp_relative_reference is not None
                    else 0.0
                )
                positions = robot.get_joint_positions()
                applied = controller.get_applied_action()
                checkpoints.append(
                    {
                        "phase": phase,
                        "elapsed_s": elapsed_s,
                        "held_object_id": latest_telemetry.held_object_id,
                        "target_position_m": [float(value) for value in target_position],
                        "target_orientation_wxyz": [float(value) for value in target_orientation],
                        "target_velocity_m_s": [
                            float(value) for value in prims[reset.target_id].get_linear_velocity()
                        ],
                        "end_effector_position_m": [float(value) for value in ee_position],
                        "end_effector_rotation_world": np.asarray(ee_rotation).tolist(),
                        "lift_height_m": float(target_position[2] - initial_target_position[2]),
                        "relative_displacement_m": slip_m,
                        "finger_positions_m": [float(positions[d.index]) for d in finger_drives],
                        "active_finger_targets_m": [
                            float(applied.joint_positions[d.index]) for d in active_fingers
                        ],
                        "raw_contact_count": latest_telemetry.raw_contact_count,
                        "contacts": [
                            {
                                "body_a": event.body_a,
                                "body_b": event.body_b,
                                "impulse_ns": event.impulse_ns,
                                "allowed": event.allowed,
                            }
                            for event in latest_telemetry.collisions
                        ],
                        "finger_sensor_readings": {
                            root: {
                                "is_valid": bool(reading.is_valid),
                                "in_contact": bool(reading.in_contact),
                                "force_n": float(reading.value),
                                "time_s": float(reading.time),
                            }
                            for root in (FRANKA_LEFT_FINGER, FRANKA_RIGHT_FINGER)
                            for reading in (sensors[root].get_sensor_reading(),)
                        },
                    }
                )
                if validate and verify_hold_during_motion:
                    require_hold(phase)
                    if slip_m > 0.015:
                        raise RuntimeError(
                            f"{phase}: target slipped {slip_m:.6f} m relative to gripper"
                        )

            def checked_action(phase: str, action) -> None:
                diagnostics["current_phase"] = phase
                try:
                    action()
                except Exception:
                    checkpoint(phase, validate=False)
                    raise
                checkpoint(phase)

            def stationary_hold() -> None:
                for _ in range(math.ceil(2.0 / physics_dt_s)):
                    step()
                    error = safety_check()
                    if error:
                        raise RuntimeError(error)

            print(
                "Running physical grasp: lift 100 mm, hold 2 s, move 50 mm, hold 2 s.", flush=True
            )
            checked_action("open", lambda: open_gripper({}))
            checked_action("align", lambda: align_over_target({}))

            checked_action("descend", lambda: descend_to_grasp({}))
            checked_action("close", lambda: close_gripper({}))
            lift_position = (plan.grasp.x_m, plan.grasp.y_m, plan.grasp.z_m + 0.10)
            checked_action("lift", lambda: move_to("grasp-test-lift", lift_position))
            if checkpoints[-1]["lift_height_m"] < task.evaluation.lift_clearance_m:
                raise RuntimeError("target did not lift clear of the ground")
            checked_action("stationary_hold", stationary_hold)
            direction = np.asarray(
                (plan.transport.x_m - plan.grasp.x_m, plan.transport.y_m - plan.grasp.y_m)
            )
            distance = float(np.linalg.norm(direction))
            if distance < 0.05:
                raise RuntimeError("basket direction is too short for the bounded lateral test")
            lateral_position = (
                lift_position[0] + float(direction[0]) * 0.05 / distance,
                lift_position[1] + float(direction[1]) * 0.05 / distance,
                lift_position[2],
            )
            checked_action("lateral_move", lambda: move_to("grasp-test-lateral", lateral_position))
            checked_action("final_hold", stationary_hold)
            output = {
                "ok": True,
                "test": "physical-grasp",
                "geometry_source": "saved-dino-sam3-rgbd-model-assisted" if perceived else "oracle-smoke-only",
                "gateway_motion_enabled": False,
                "gripper_preload": gripper_preload,
                "wrist_drive": wrist_drive,
                "maximum_relative_displacement_m": maximum_grasp_slip_m,
                "elapsed_s": elapsed_s,
            }
            write_result(output)
            print(json.dumps({key: value for key, value in output.items() if key != "diagnostics"},
                             indent=2, sort_keys=True), flush=True)
            return 0

        actions = {
            "open_gripper": open_gripper,
            "align_over_target": align_over_target,
            "descend_to_grasp": descend_to_grasp,
            "close_gripper": close_gripper,
            "lift_grasped": lift_grasped,
            "move_over_basket": move_over_basket,
            "position_for_release": position_for_release,
            "release_object": release_object,
            "retract_from_basket": retract_from_basket,
            "settle_release": settle_release,
        }
        print("Executing bounded perceived-geometry graph." if perceived else
              "Executing bounded oracle-geometry reference graph.", flush=True)
        graph_result = GraphPolicyExecutor(graph, actions).execute(
            {"seed": reset.seed, "target_id": reset.target_id}
        )

        if graph_result.status != "succeeded" and not latest_evaluation.terminal:
            failure_detail = graph_result.trace[-1].detail if graph_result.trace else "graph failed"
            latest_evaluation = evaluator.update(
                episode_runtime.capture_frame(
                    elapsed_s,
                    held_object_id=latest_telemetry.held_object_id,
                    collisions=latest_telemetry.collisions,
                    graph_error=failure_detail,
                )
            )
        ok = (
            graph_result.status == "succeeded"
            and latest_evaluation.status is EpisodeStatus.SUCCEEDED
        )
        output = {
            "ok": ok,
            "geometry_source": graph.metadata["geometry_source"],
            "graph_authoring": graph_authoring,
            "graph_sha256": graph_hash,
            "generation_receipt": generation_receipt,
            "maximum_relative_displacement_m": maximum_grasp_slip_m,
            "grip_material": {"static_friction": 1.2, "dynamic_friction": 1.0},
            "wrist_drive": wrist_drive,
            "gripper_preload": gripper_preload,
            "motion_limits": {
                "joint_tolerance_rad": motion_policy.final_position_tolerance,
                "minimum_settle_time_s": motion_policy.final_settle_time_s,
                "maximum_settle_time_s": motion_policy.max_final_settle_time_s,
            },
            "gateway_motion_enabled": False,
            "graph": graph_result.as_dict(),
            "evaluation": latest_evaluation.as_dict(),
            "elapsed_s": elapsed_s,
        }
        write_result(output)
        print(json.dumps({key: value for key, value in output.items() if key != "diagnostics"},
                         indent=2, sort_keys=True), flush=True)
        return 0 if ok else 1
    except Exception as error:  # noqa: BLE001 - report before Isaac shutdown
        write_result({"ok": False, "error_type": type(error).__name__, "error": str(error)})
        print(
            f"Graph pick-and-place validation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 1
    finally:
        if descent_trace_stream is not None:
            descent_trace_stream.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
