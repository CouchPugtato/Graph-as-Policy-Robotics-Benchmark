"""Build a deterministic grocery-fulfillment proxy scene in Isaac Sim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing stage: {output}")
    if output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise SystemExit("--output must end in .usd, .usda, or .usdc")
    if not args.task.is_file():
        raise SystemExit(f"Task configuration does not exist: {args.task}")
    output.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gap_isaac.camera_config import (
        CAPTURE_RESOLUTION,
        GROCERY_CAMERA_EYE_M,
        GROCERY_CAMERA_TARGET_M,
        apply_pinhole_lens,
        build_pinhole_lens,
    )
    from gap_isaac.config import load_task
    from gap_isaac.grocery_scene import (
        GROUND_POSITION_M,
        GROUND_PRIM_PATH,
        GROUND_SIZE_M,
        ROBOT_BASE_POSITION_M,
        SCENE_LAYOUT,
        WORK_SURFACE_Z_M,
        build_grocery_scene_plan,
    )

    task = load_task(args.task)
    plan = build_grocery_scene_plan(task, seed=args.seed, target_id=args.target)

    # Isaac modules must only be imported after SimulationApp is created.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True, "width": 640, "height": 480})
    try:
        import numpy as np
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.experimental.utils.semantics import add_labels
        from isaacsim.core.utils.stage import add_reference_to_stage, save_stage
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

        def metadata(prim: Any, object_id: str, semantic_label: str) -> None:
            prim.CreateAttribute("gap:objectId", Sdf.ValueTypeNames.String).Set(object_id)
            add_labels(prim, labels=[semantic_label], taxonomy="class")

        def cube(
            path: str,
            position: tuple[float, float, float],
            size: tuple[float, float, float],
            color: tuple[float, float, float],
            *,
            yaw_deg: float = 0.0,
            rigid: bool = False,
            mass_kg: float = 0.0,
        ) -> Any:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
            geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            xform = UsdGeom.Xformable(geom.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*position))
            xform.AddRotateZOp().Set(yaw_deg)
            xform.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            if rigid:
                UsdPhysics.RigidBodyAPI.Apply(geom.GetPrim())
                UsdPhysics.MassAPI.Apply(geom.GetPrim()).CreateMassAttr(mass_kg)
            return geom.GetPrim()

        def cylinder(
            path: str,
            position: tuple[float, float, float],
            size: tuple[float, float, float],
            color: tuple[float, float, float],
            yaw_deg: float,
            mass_kg: float,
        ) -> Any:
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateAxisAttr("Z")
            geom.CreateRadiusAttr(size[0] / 2)
            geom.CreateHeightAttr(size[2])
            geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            xform = UsdGeom.Xformable(geom.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*position))
            xform.AddRotateZOp().Set(yaw_deg)
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(geom.GetPrim())
            UsdPhysics.MassAPI.Apply(geom.GetPrim()).CreateMassAttr(mass_kg)
            return geom.GetPrim()

        World(stage_units_in_meters=1.0)
        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        stage.SetDefaultPrim(world_prim)
        stage.DefinePrim("/World/Task", "Xform")

        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError("Isaac asset root is unavailable; check internet or asset settings")
        franka_asset = f"{assets_root}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        franka_prim = add_reference_to_stage(franka_asset, "/World/Franka")
        franka_xform = UsdGeom.Xformable(franka_prim)
        translate_ops = [
            operation
            for operation in franka_xform.GetOrderedXformOps()
            if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        translate_op = translate_ops[0] if translate_ops else franka_xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*ROBOT_BASE_POSITION_M))

        # One static floor; both the robot mount and basket bottom are at z=0.
        # Dynamic groceries start 3 mm above it, then settle under gravity.
        cube(
            GROUND_PRIM_PATH,
            GROUND_POSITION_M,
            GROUND_SIZE_M,
            (0.50, 0.50, 0.50),
        )

        for item in plan.objects:
            path = f"/World/Task/{item.object_id}"
            if item.is_container:
                parent = stage.DefinePrim(path, "Xform")
                xform = UsdGeom.Xformable(parent)
                xform.AddTranslateOp().Set(Gf.Vec3d(*item.position_m))
                xform.AddRotateZOp().Set(item.yaw_deg)
                metadata(parent, item.object_id, item.semantic_label)
                width, depth, height = item.size_m
                wall = task.evaluation.basket_wall_thickness_m
                base = task.evaluation.basket_base_thickness_m
                cube(f"{path}/base", (0, 0, base / 2), (width, depth, base), item.color_rgb)
                cube(
                    f"{path}/left",
                    (-width / 2 + wall / 2, 0, base + height / 2),
                    (wall, depth, height),
                    item.color_rgb,
                )
                cube(
                    f"{path}/right",
                    (width / 2 - wall / 2, 0, base + height / 2),
                    (wall, depth, height),
                    item.color_rgb,
                )
                cube(
                    f"{path}/front",
                    (0, -depth / 2 + wall / 2, base + height / 2),
                    (width - 2 * wall, wall, height),
                    item.color_rgb,
                )
                cube(
                    f"{path}/back",
                    (0, depth / 2 - wall / 2, base + height / 2),
                    (width - 2 * wall, wall, height),
                    item.color_rgb,
                )
            elif item.shape == "cylinder":
                prim = cylinder(
                    path,
                    item.position_m,
                    item.size_m,
                    item.color_rgb,
                    item.yaw_deg,
                    item.mass_kg,
                )
                metadata(prim, item.object_id, item.semantic_label)
            else:
                prim = cube(
                    path,
                    item.position_m,
                    item.size_m,
                    item.color_rgb,
                    yaw_deg=item.yaw_deg,
                    rigid=True,
                    mass_kg=item.mass_kg,
                )
                metadata(prim, item.object_id, item.semantic_label)

        evaluation = stage.DefinePrim("/World/Evaluation", "Scope")
        evaluation.CreateAttribute("gap:seed", Sdf.ValueTypeNames.Int).Set(plan.seed)
        evaluation.CreateAttribute("gap:targetId", Sdf.ValueTypeNames.String).Set(plan.target_id)
        evaluation.CreateAttribute("gap:containerId", Sdf.ValueTypeNames.String).Set(
            task.container_id
        )
        evaluation.CreateAttribute("gap:sceneLayout", Sdf.ValueTypeNames.String).Set(SCENE_LAYOUT)
        evaluation.CreateAttribute("gap:workSurfaceZ", Sdf.ValueTypeNames.Double).Set(
            WORK_SURFACE_Z_M
        )
        evaluation.CreateAttribute("gap:robotBaseZ", Sdf.ValueTypeNames.Double).Set(
            ROBOT_BASE_POSITION_M[2]
        )
        evaluation.CreateAttribute("gap:liftClearanceM", Sdf.ValueTypeNames.Double).Set(
            task.evaluation.lift_clearance_m
        )
        evaluation.CreateAttribute("gap:containmentMarginM", Sdf.ValueTypeNames.Double).Set(
            task.evaluation.containment_margin_m
        )
        evaluation.CreateAttribute("gap:maxReleaseSpeedMps", Sdf.ValueTypeNames.Double).Set(
            task.evaluation.max_release_speed_m_s
        )
        collision_threshold = evaluation.CreateAttribute(
            "gap:collisionImpulseThresholdNs",
            Sdf.ValueTypeNames.Double,
        )
        collision_threshold.Set(task.evaluation.collision_impulse_threshold_ns)
        evaluation.CreateAttribute("gap:graspContactFrames", Sdf.ValueTypeNames.Int).Set(
            task.evaluation.grasp_contact_frames
        )
        evaluation.CreateAttribute("gap:releaseContactFrames", Sdf.ValueTypeNames.Int).Set(
            task.evaluation.release_contact_frames
        )

        Camera(
            prim_path="/World/Camera",
            position=np.array(GROCERY_CAMERA_EYE_M),
            resolution=CAPTURE_RESOLUTION,
        )
        # Author a wide, square-pixel lens instead of USD's narrow default.
        apply_pinhole_lens(UsdGeom.Camera.Get(stage, "/World/Camera"), build_pinhole_lens())
        set_camera_view(
            eye=np.array(GROCERY_CAMERA_EYE_M),
            target=np.array(GROCERY_CAMERA_TARGET_M),
            camera_prim_path="/World/Camera",
        )
        light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        light.CreateIntensityAttr(900.0)

        for _ in range(3):
            simulation_app.update()
        actual_robot_base = franka_xform.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ).ExtractTranslation()
        if any(
            abs(float(actual_robot_base[index]) - expected) > 1e-6
            for index, expected in enumerate(ROBOT_BASE_POSITION_M)
        ):
            raise RuntimeError(
                f"Franka mount transform is {tuple(actual_robot_base)}, "
                f"expected {ROBOT_BASE_POSITION_M}"
            )
        required = ("/World/Franka", "/World/Franka/panda_hand", "/World/Camera")
        missing = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
        if missing:
            raise RuntimeError(f"created stage is missing required prims: {', '.join(missing)}")
        if stage.GetPrimAtPath("/Render").IsValid():
            stage.RemovePrim("/Render")
        if not save_stage(str(output), save_and_reload_in_place=False):
            raise RuntimeError(f"Isaac could not save stage: {output}")
        print(
            f"Created ground-level grocery scene: {output} "
            f"(seed={plan.seed}, target={plan.target_id}, layout={SCENE_LAYOUT})"
        )
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
