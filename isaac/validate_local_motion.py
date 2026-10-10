"""Run one minimal, collision-monitored local motion in native Isaac Sim."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", default="alphabet_soup")
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--gripper-contact", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.stage.is_file():
        raise SystemExit(f"USD stage does not exist: {args.stage}")
    if not args.task.is_file():
        raise SystemExit(f"Task configuration does not exist: {args.task}")

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from isaacsim import SimulationApp

    def write_result(payload: dict) -> None:
        if args.result_file is not None:
            args.result_file.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    simulation_app = SimulationApp({"headless": args.headless, "width": 640, "height": 480})
    try:
        import numpy as np
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.experimental.physics import Contact, ContactSensor
        from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics

        from gap_isaac.config import load_task
        from gap_isaac.contact_telemetry import (
            FRANKA_CONTACT_LINKS,
            FRANKA_GROUND_MOUNT_CONTACTS,
            FRANKA_LEFT_FINGER,
            FRANKA_RIGHT_FINGER,
            ContactInterpreter,
            IsaacContactCollector,
        )
        from gap_isaac.grocery_scene import GEOMETRY, ROBOT_BASE_POSITION_M, validate_ground_scene
        from gap_isaac.isaac_episode import IsaacEpisodeRuntime
        from gap_isaac.local_gripper import LocalGripperExecutor
        from gap_isaac.local_motion import LocalTrajectoryExecutor
        from gap_isaac.protocol import ExecuteTrajectoryRequest, TrajectoryPoint
        from gap_isaac.reset import build_episode_reset_plan

        if not open_stage(str(args.stage.resolve())):
            raise RuntimeError(f"Isaac could not open USD stage: {args.stage}")
        validate_ground_scene(omni.usd.get_context().get_stage())
        task = load_task(args.task)
        reset = build_episode_reset_plan(task, seed=args.seed, target_id=args.target)
        world = World(stage_units_in_meters=1.0)
        robot = SingleArticulation("/World/Franka", name="motion_franka")
        world.scene.add(robot)
        prims = {}
        for state in reset.objects:
            path = f"/World/Task/{state.object_id}"
            if state.object_id == reset.container_id:
                prim = SingleXFormPrim(path, name=f"motion_{state.object_id}")
            else:
                prim = SingleRigidPrim(path, name=f"motion_{state.object_id}")
            prims[state.object_id] = prim
            world.scene.add(prim)

        stage = omni.usd.get_context().get_stage()
        missing_links = [
            path for path in FRANKA_CONTACT_LINKS if not stage.GetPrimAtPath(path).IsValid()
        ]
        if missing_links:
            raise RuntimeError(f"Franka stage is missing contact links: {missing_links}")
        sensors = {
            link_path: ContactSensor(
                Contact.create(
                    f"{link_path}/gap_motion_contact_sensor",
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
            grasp_contact_frames=task.evaluation.grasp_contact_frames,
            release_contact_frames=task.evaluation.release_contact_frames,
        )
        contacts = IsaacContactCollector(
            sensors,
            lambda body_id: str(PhysicsSchemaTools.intToSdfPath(body_id)),
            interpreter,
        )

        if args.gripper_contact:
            # The gripper has only 5 mm of clearance per side around the
            # 70 mm can when fully open. PhysX's general-purpose default
            # contact margins are wider than that clearance, so use a narrow
            # runtime-only margin for this controlled contact fixture.
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

        world.reset()
        robot_base_position, _ = robot.get_world_pose()
        robot_base_position = tuple(float(value) for value in robot_base_position)
        print(f"Live Franka base position: {robot_base_position}", flush=True)
        if any(
            abs(robot_base_position[index] - expected) > 1e-3
            for index, expected in enumerate(ROBOT_BASE_POSITION_M)
        ):
            raise RuntimeError(
                f"live Franka base is {robot_base_position}, expected {ROBOT_BASE_POSITION_M}"
            )
        runtime = IsaacEpisodeRuntime(robot, prims, reset)
        runtime.apply_reset()
        contacts.reset()
        physics_dt_s = float(world.get_physics_dt())
        for _ in range(runtime.settle_step_count(physics_dt_s)):
            world.step(render=False)
        if contacts.capture().held_object_id is not None:
            raise RuntimeError("gripper falsely reports a held object before motion")

        def action_factory(targets, indices):
            return ArticulationAction(
                joint_positions=np.asarray(targets, dtype=np.float64),
                joint_indices=np.asarray(indices, dtype=np.int64),
            )

        if args.gripper_contact:
            gripper = LocalGripperExecutor(
                robot,
                physics_dt_s=physics_dt_s,
                step=lambda: world.step(render=False),
                action_factory=action_factory,
                contact_capture=contacts.capture,
            )
            close_result = gripper.move("free-space-close", open_gripper=False)
            if close_result.status != "succeeded":
                raise RuntimeError(
                    f"free-space close failed: {close_result.status}: {close_result.detail}"
                )
            open_result = gripper.move("free-space-open", open_gripper=True)
            if open_result.status != "succeeded":
                raise RuntimeError(
                    f"free-space open failed: {open_result.status}: {open_result.detail}"
                )

            hand = SingleXFormPrim("/World/Franka/panda_hand", name="gripper_test_hand")
            hand_position, hand_orientation = hand.get_world_pose()
            w, x, y, z = (float(value) for value in hand_orientation)
            forward = np.asarray(
                [
                    2 * (x * z + w * y),
                    2 * (y * z - w * x),
                    1 - 2 * (x * x + y * y),
                ],
                dtype=np.float64,
            )
            # The finger joints begin 58.4 mm in front of the hand frame. Keep
            # the object's near face beyond the palm while leaving it inside
            # the finger contact region.
            fixture_axial_offset_m = 0.075 + GEOMETRY[reset.target_id].size_m[2] / 2
            fixture_position = (
                np.asarray(hand_position, dtype=np.float64) + fixture_axial_offset_m * forward
            )
            target_prim = prims[reset.target_id]
            target_usd_prim = stage.GetPrimAtPath(f"/World/Task/{reset.target_id}")
            rigid_api = UsdPhysics.RigidBodyAPI(target_usd_prim)
            rigid_api.CreateKinematicEnabledAttr().Set(True)
            target_prim.set_world_pose(
                position=fixture_position,
                orientation=np.asarray(hand_orientation, dtype=np.float64),
            )
            contacts.reset()
            fixture_telemetry = None
            for _ in range(3):
                world.step(render=False)
                fixture_telemetry = contacts.capture()
            if fixture_telemetry is None or fixture_telemetry.held_object_id is not None:
                raw_contact_count = fixture_telemetry.raw_contact_count if fixture_telemetry else 0
                collision_summary = [
                    (f"{event.body_a} with {event.body_b} ({event.impulse_ns:.6g} N*s)")
                    for event in (fixture_telemetry.collisions if fixture_telemetry else ())
                ]
                raise RuntimeError(
                    "open gripper fixture must begin without a verified hold; "
                    f"separation={gripper.separation_m():.6f} m, "
                    f"raw_contacts={raw_contact_count}, "
                    f"contacts={collision_summary}"
                )

            grasp_result = gripper.move(
                "controlled-contact-close",
                open_gripper=False,
                require_held_object_id=reset.target_id,
            )
            if grasp_result.status != "succeeded":
                raise RuntimeError(
                    f"controlled contact failed: {grasp_result.status}: {grasp_result.detail}"
                )
            grasp_separation_m = gripper.separation_m()
            release_result = gripper.move("controlled-contact-open", open_gripper=True)
            if release_result.status != "succeeded":
                raise RuntimeError(
                    f"controlled release failed: {release_result.status}: {release_result.detail}"
                )
            release_telemetry = contacts.capture()
            for _ in range(task.evaluation.release_contact_frames + 2):
                if release_telemetry.held_object_id is None:
                    break
                world.step(render=False)
                release_telemetry = contacts.capture()
            if release_telemetry.held_object_id is not None:
                raise RuntimeError(
                    f"held-object signal did not clear: {release_telemetry.held_object_id}"
                )

            output = {
                "ok": True,
                "free_space_cycle": "succeeded",
                "controlled_contact": "succeeded",
                "verified_held_object_id": reset.target_id,
                "grasp_separation_m": grasp_separation_m,
                "release_verified": True,
                "gateway_motion_enabled": False,
            }
            write_result(output)
            print(json.dumps(output, indent=2, sort_keys=True), flush=True)
            return 0
        print("Reset settled; executing bounded local motion.", flush=True)

        monitored_frames = 0
        max_disallowed_impulse_ns = 0.0

        def safety_check() -> str | None:
            nonlocal monitored_frames, max_disallowed_impulse_ns
            monitored_frames += 1
            telemetry = contacts.capture()
            disallowed = [
                event
                for event in telemetry.collisions
                if not event.allowed
                and event.impulse_ns >= task.evaluation.collision_impulse_threshold_ns
            ]
            if not disallowed:
                return None
            strongest = max(disallowed, key=lambda event: event.impulse_ns)
            max_disallowed_impulse_ns = max(max_disallowed_impulse_ns, strongest.impulse_ns)
            return (
                f"disallowed collision: {strongest.body_a} with {strongest.body_b} "
                f"({strongest.impulse_ns:.6f} N*s)"
            )

        executor = LocalTrajectoryExecutor(
            robot,
            physics_dt_s=physics_dt_s,
            step=lambda: world.step(render=False),
            action_factory=action_factory,
            safety_check=safety_check,
            allowed_joint_names=reset.robot_joint_names[:7],
        )
        home_joint1 = reset.robot_joint_positions_rad[0]
        request = ExecuteTrajectoryRequest(
            command_id="local-joint1-round-trip",
            joint_names=(reset.robot_joint_names[0],),
            points=(
                TrajectoryPoint(
                    time_from_start_s=1.0,
                    positions_rad=(home_joint1 + 0.05,),
                ),
                TrajectoryPoint(time_from_start_s=2.0, positions_rad=(home_joint1,)),
            ),
            timeout_s=3.0,
        )
        result = executor.execute(request)
        if result.status != "succeeded":
            raise RuntimeError(f"local motion failed: {result.status}: {result.detail}")

        output = {
            "ok": True,
            "command_id": result.command_id,
            "command_status": result.status,
            "gateway_motion_enabled": False,
            "joint": reset.robot_joint_names[0],
            "maximum_offset_rad": 0.05,
            "monitored_frames": monitored_frames,
            "max_disallowed_impulse_ns": max_disallowed_impulse_ns,
            "returned_home": True,
        }
        write_result(output)
        print(json.dumps(output, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:  # noqa: BLE001 - report before Isaac shutdown
        write_result(
            {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        print(
            f"Local motion validation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
