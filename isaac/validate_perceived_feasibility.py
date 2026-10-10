"""Finite Isaac alignment/IK diagnostic. Does not apply actions or save the stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--target-geometry", required=True, type=Path)
    parser.add_argument("--basket-geometry", required=True, type=Path)
    parser.add_argument("--result-file", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from gap_isaac.diagnostic_progress import DiagnosticProgress
    progress = DiagnosticProgress(args.result_file.with_suffix(".progress.json"))
    progress.mark("load_saved_inputs")
    from gap_isaac.config import load_task
    from gap_isaac.perceived_feasibility import prepare_checks
    from gap_isaac.observation_bundle import ObservationBundle
    target = json.loads(args.target_geometry.read_text())
    basket = json.loads(args.basket_geometry.read_text())
    task = load_task(root / "configs/tasks/grocery_fulfillment.yaml")
    checks = prepare_checks(target, basket, task.evaluation.basket_wall_thickness_m,
                            task.evaluation.containment_margin_m)
    bundle = ObservationBundle.load(Path(target["bundle"]))
    from isaacsim import SimulationApp
    progress.mark("simulation_app_start")
    app = SimulationApp({"headless": True, "width": 320, "height": 240,
                         "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False,
                         "limit_cpu_threads": 2,
                         "extra_args": ["--enable", "isaacsim.robot_motion.motion_generation"]})
    progress.mark("simulation_app_ready")
    result = {"ok": False, "checks": checks, "commands_applied": 0,
              "graph_motion_ready": False, "stage_saved": False}
    code = 1
    try:
        import numpy as np
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.sensors.camera import Camera
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, load_supported_lula_kinematics_solver_config
        from pxr import UsdGeom
        from gap_isaac.grocery_scene import validate_ground_scene
        progress.mark("open_stage_start", path=str(args.stage))
        if not open_stage(str(args.stage.resolve())):
            raise RuntimeError("Could not open requested ground stage")
        progress.mark("open_stage_done")
        stage = omni.usd.get_context().get_stage()
        validate_ground_scene(stage)
        world = World(stage_units_in_meters=1, physics_dt=1 / 60, rendering_dt=1 / 60)
        robot = world.scene.add(SingleArticulation("/World/Franka", name="check_franka"))
        target_prim = world.scene.add(SingleRigidPrim("/World/Task/alphabet_soup", name="check_target"))
        basket_prim = world.scene.add(SingleXFormPrim("/World/Task/basket", name="check_basket"))
        # No camera initialization/render product/model is needed to read pose.
        camera = Camera("/World/Camera", resolution=(640, 480))
        progress.mark("world_reset_start")
        world.reset()
        progress.mark("world_reset_done")
        # Exactly two simulated seconds of passive settling; no control actions.
        progress.mark("passive_settle_start", total_frames=120)
        for frame in range(120):
            world.step(render=False)
            if (frame + 1) % 30 == 0:
                progress.mark("passive_settle_progress", completed_frames=frame + 1)
        progress.mark("read_live_alignment")
        live_target, _ = target_prim.get_world_pose()
        live_basket, _ = basket_prim.get_world_pose()
        base_position, base_orientation = robot.get_world_pose()
        camera_position, camera_orientation = camera.get_world_pose(camera_axes="ros")
        saved_pose = bundle.observation.camera_pose
        saved_position = np.array([saved_pose.position.x, saved_pose.position.y, saved_pose.position.z])
        q = saved_pose.orientation
        saved_orientation = np.array([q.w, q.x, q.y, q.z])
        camera_position_error = float(np.linalg.norm(camera_position - saved_position))
        camera_angle = float(2 * np.arccos(np.clip(abs(np.dot(camera_orientation, saved_orientation)), 0, 1)))
        usd_camera = UsdGeom.Camera.Get(stage, "/World/Camera")
        fx = 640 * usd_camera.GetFocalLengthAttr().Get() / usd_camera.GetHorizontalApertureAttr().Get()
        focal_error = abs(float(fx) - bundle.observation.camera.fx)
        target_center = np.array([target["obb"]["center"][axis] for axis in ("x", "y", "z")])
        basket_center = np.array([basket["obb"]["center"][axis] for axis in ("x", "y", "z")])
        target_error = float(np.linalg.norm(target_center - live_target))
        basket_error_xy = float(np.linalg.norm(basket_center[:2] - live_basket[:2]))
        result["alignment"] = {"live_target_position_m": live_target.tolist(),
                               "live_basket_origin_m": live_basket.tolist(),
                               "target_center_error_m": target_error,
                               "basket_center_xy_error_m": basket_error_xy,
                               "camera_position_error_m": camera_position_error,
                               "camera_orientation_error_rad": camera_angle,
                               "camera_focal_error_px": focal_error,
                               "robot_base_position_m": base_position.tolist(),
                               "passive_settle_frames": 120,
                               "comparison_source": "privileged live simulator state, diagnostics only"}
        alignment_ok = (target_error <= .02 and basket_error_xy <= .025
                        and camera_position_error <= .001 and camera_angle <= .001 and focal_error <= 1)
        result["alignment"]["passed"] = alignment_ok
        progress.mark("alignment_done", passed=bool(alignment_ok))
        progress.mark("lula_initialize_start")
        lula = LulaKinematicsSolver(**load_supported_lula_kinematics_solver_config("Franka"))
        lula.set_robot_base_pose(base_position, base_orientation)
        lula.sampling_seed = 0
        names = lula.get_joint_names()
        current = robot.get_joint_positions()
        warm = np.array([current[list(robot.dof_names).index(name)] for name in names])
        desired_orientation = np.array(checks["orientation_wxyz"], dtype=np.float64)
        progress.mark("lula_initialize_done")
        solutions = {}
        for name, position in checks["waypoints_m"].items():
            progress.mark("ik_start", waypoint=name)
            joints, solved = lula.compute_inverse_kinematics("right_gripper", np.array(position),
                                desired_orientation, warm_start=warm,
                                position_tolerance=.005, orientation_tolerance=.05)
            entry = {"solved": bool(solved), "target_position_m": position}
            if solved:
                actual, rotation = lula.compute_forward_kinematics("right_gripper", joints)
                entry.update(joint_names=names, joint_positions_rad=joints.tolist(),
                             fk_position_error_m=float(np.linalg.norm(np.asarray(actual).reshape(3) - position)))
                warm = joints
            solutions[name] = entry
            progress.mark("ik_done", waypoint=name, solved=bool(solved))
        result["ik"] = solutions
        result["grasp_pose_ik_feasible"] = all(item["solved"] and item.get("fk_position_error_m", 1) <= .005
                                               for item in solutions.values())
        result["ok"] = bool(alignment_ok and result["grasp_pose_ik_feasible"]
                            and checks["opening_fit_model_assisted"] and checks["finger_aperture_fit"])
        result["collision_free_path_validated"] = False
        result["physical_hold_validated"] = False
        result["capture_freshness_note"] = "Existing saved capture checked against this finite stage instance; recheck at execution."
        code = 0 if result["ok"] else 1
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        progress.mark("diagnostic_error", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        progress.mark("write_result", ok=result["ok"])
        args.result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        progress.mark("simulation_shutdown_start")
        app.close()
        progress.mark("simulation_shutdown_done")
    print(json.dumps({"ok": result["ok"], "result": str(args.result_file),
                      "graph_motion_ready": False, "commands_applied": 0}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
