"""Create the smallest Franka scene required by the read-only connector.

Run this file with Isaac Sim's python.bat. It creates one camera, one Franka,
and one light, saves the USD, and exits without starting a long-lived server.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing stage: {output}")
    if output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise SystemExit("--output must end in .usd, .usda, or .usdc")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Isaac modules must only be imported after SimulationApp is created.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True, "width": 640, "height": 480})
    try:
        import numpy as np
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage, save_stage
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path
        from pxr import UsdLux

        World(stage_units_in_meters=1.0)
        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError("Isaac asset root is unavailable; check internet or asset settings")

        franka_asset = f"{assets_root}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        add_reference_to_stage(franka_asset, "/World/Franka")
        Camera(
            prim_path="/World/Camera",
            position=np.array([1.5, 1.5, 1.2]),
            resolution=(640, 480),
        )
        set_camera_view(
            eye=np.array([1.5, 1.5, 1.2]),
            target=np.array([0.25, 0.0, 0.45]),
            camera_prim_path="/World/Camera",
        )

        stage = omni.usd.get_context().get_stage()
        light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        light.CreateIntensityAttr(750.0)

        # Give the referenced robot a few update cycles to compose before validation.
        for _ in range(3):
            simulation_app.update()
        required_prims = ("/World/Franka", "/World/Franka/panda_hand", "/World/Camera")
        missing = [path for path in required_prims if not stage.GetPrimAtPath(path).IsValid()]
        if missing:
            raise RuntimeError(f"created stage is missing required prims: {', '.join(missing)}")

        # Render products are runtime objects. Keeping them in the USD causes
        # duplicate annotators and preserves the builder's low preview resolution.
        if stage.GetPrimAtPath("/Render").IsValid():
            stage.RemovePrim("/Render")
        if not save_stage(str(output), save_and_reload_in_place=False):
            raise RuntimeError(f"Isaac could not save stage: {output}")
        print(f"Created read-only connector stage: {output}")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
