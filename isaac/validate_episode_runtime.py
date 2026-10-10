"""Apply and validate one deterministic reset inside native Isaac Sim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", default="alphabet_soup")
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

    simulation_app = SimulationApp({"headless": args.headless, "width": 640, "height": 480})
    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.sensors.experimental.physics import Contact, ContactSensor
        from pxr import PhysicsSchemaTools

        from gap_isaac.config import load_task
        from gap_isaac.contact_telemetry import (
            FRANKA_CONTACT_LINKS,
            FRANKA_GROUND_MOUNT_CONTACTS,
            FRANKA_LEFT_FINGER,
            FRANKA_RIGHT_FINGER,
            ContactInterpreter,
            IsaacContactCollector,
        )
        from gap_isaac.evaluate import PrivilegedEpisodeEvaluator
        from gap_isaac.grocery_scene import validate_ground_scene
        from gap_isaac.isaac_episode import IsaacEpisodeRuntime
        from gap_isaac.reset import build_episode_reset_plan

        if not open_stage(str(args.stage.resolve())):
            raise RuntimeError(f"Isaac could not open USD stage: {args.stage}")
        validate_ground_scene(omni.usd.get_context().get_stage())

        task = load_task(args.task)
        reset = build_episode_reset_plan(task, seed=args.seed, target_id=args.target)
        world = World(stage_units_in_meters=1.0)
        robot = SingleArticulation("/World/Franka", name="episode_franka")
        world.scene.add(robot)
        prims = {}
        for state in reset.objects:
            path = f"/World/Task/{state.object_id}"
            if state.object_id == reset.container_id:
                prim = SingleXFormPrim(path, name=f"episode_{state.object_id}")
            else:
                prim = SingleRigidPrim(path, name=f"episode_{state.object_id}")
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
                    f"{link_path}/gap_contact_sensor",
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

        world.reset()
        runtime = IsaacEpisodeRuntime(robot, prims, reset)
        runtime.apply_reset()
        contacts.reset()
        for _ in range(runtime.settle_step_count(world.get_physics_dt())):
            world.step(render=False)

        contact_frame = contacts.capture()
        if contact_frame.held_object_id is not None:
            raise RuntimeError(
                f"freshly reset gripper falsely reports a hold: {contact_frame.held_object_id}"
            )
        frame = runtime.capture_frame(
            reset.settle_time_s,
            held_object_id=contact_frame.held_object_id,
            collisions=contact_frame.collisions,
        )
        result = PrivilegedEpisodeEvaluator(task, reset).update(frame)
        output = {
            "seed": reset.seed,
            "target_id": reset.target_id,
            "status": result.status.value,
            "object_count": len(frame.objects),
            "contact_sensor_count": len(sensors),
            "raw_contact_count": contact_frame.raw_contact_count,
            "held_object_id": contact_frame.held_object_id,
            "settle_time_s": reset.settle_time_s,
            "motion_enabled": False,
        }
        print(json.dumps(output, indent=2, sort_keys=True), flush=True)
        if result.terminal:
            detail = result.as_dict()
            raise RuntimeError(f"freshly reset episode is unexpectedly terminal: {detail}")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
