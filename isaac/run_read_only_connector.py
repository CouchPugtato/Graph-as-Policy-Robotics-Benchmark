"""Run the observation-only connector inside Isaac Sim's Python runtime.

This script attaches to an existing USD stage. It cannot execute trajectories
or operate the gripper. Run it with Isaac Sim's python.bat, never system Python.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--robot-prim", default="/World/Franka")
    parser.add_argument("--camera-prim", default="/World/Camera")
    parser.add_argument("--end-effector-prim", default="/World/Franka/panda_hand")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--capture-every", type=int, default=6)
    parser.add_argument("--horizontal-fov-deg", type=float, default=65.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def run_gateway_server(server: Any) -> None:
    """Run Uvicorn without its asyncio.run wrapper.

    Isaac Sim patches asyncio.run with a signature that is incompatible with
    recent Uvicorn releases. A thread-owned event loop avoids that global
    wrapper while keeping all HTTP work off Isaac's simulation thread.
    """

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        asyncio.set_event_loop(None)
        loop.close()


def configure_long_running_timeline(timeline: Any) -> None:
    """Prevent a short USD timeline from terminating the connector."""

    timeline.set_end_time(1_000_000.0)
    timeline.set_looping(True)


def main() -> int:
    args = parse_args()
    if not args.stage.is_file():
        raise SystemExit(f"USD stage does not exist: {args.stage}")
    if args.capture_every < 1:
        raise SystemExit("--capture-every must be positive")
    if not math.isfinite(args.horizontal_fov_deg) or not 30 <= args.horizontal_fov_deg <= 90:
        raise SystemExit("--horizontal-fov-deg must be between 30 and 90")

    token = os.environ.get("GAP_ISAAC_CONNECTOR_TOKEN")
    if not token:
        raise SystemExit("GAP_ISAAC_CONNECTOR_TOKEN must be set")

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    # Isaac modules must only be imported after SimulationApp is created.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless, "width": 640, "height": 480})
    server = None
    server_thread = None
    snapshots = None
    try:
        import omni.timeline
        import omni.usd
        import uvicorn
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.sensors.camera import Camera
        from pxr import Usd, UsdGeom

        from gap_isaac.camera_config import (
            CAPTURE_RESOLUTION,
            apply_pinhole_lens,
            build_pinhole_lens,
        )
        from gap_isaac.gateway import create_app
        from gap_isaac.isaac_backend import IsaacReadOnlyBackend, SnapshotBuffer
        from gap_isaac.isaac_sim_adapter import IsaacSimObservationAdapter

        if not open_stage(str(args.stage.resolve())):
            raise RuntimeError(f"Isaac could not open USD stage: {args.stage}")

        # The scene builder uses a Camera helper to author its pose. That can
        # persist a temporary render product, which must not be reused here.
        stage = omni.usd.get_context().get_stage()
        stage.SetStartTimeCode(0.0)
        stage.SetEndTimeCode(60_000_000.0)
        if stage.GetPrimAtPath("/Render").IsValid():
            stage.RemovePrim("/Render")

        # Apply to existing stages without rebuilding or saving over them.
        lens = build_pinhole_lens(args.horizontal_fov_deg)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            apply_pinhole_lens(UsdGeom.Camera.Get(stage, args.camera_prim), lens)
        print(
            f"Connector camera: {lens.width}x{lens.height}, "
            f"horizontal FOV={lens.horizontal_fov_deg:.1f} deg, "
            f"expected fx/fy={lens.focal_pixels:.3f} px (session-only lens override).",
            flush=True,
        )

        world = World(stage_units_in_meters=1.0)
        robot = SingleArticulation(args.robot_prim, name="gap_franka")
        camera = Camera(args.camera_prim, name="gap_camera", resolution=CAPTURE_RESOLUTION)
        end_effector = SingleXFormPrim(args.end_effector_prim, name="gap_end_effector")

        world.scene.add(robot)
        world.scene.add(camera)
        world.reset()
        configure_long_running_timeline(omni.timeline.get_timeline_interface())
        camera.add_distance_to_image_plane_to_frame()

        adapter = IsaacSimObservationAdapter(camera, robot, end_effector)
        snapshots = SnapshotBuffer()
        backend = IsaacReadOnlyBackend(
            snapshots,
            base_url=f"http://{args.host}:{args.port}",
            retained_frames=2,
        )
        app = create_app(backend, token)
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        server_thread = threading.Thread(
            target=run_gateway_server,
            args=(server,),
            daemon=True,
            name="connector-gateway",
        )
        server_thread.start()
        print(f"Connector gateway starting on http://{args.host}:{args.port}", flush=True)

        sequence = 0
        ticks = 0
        stop_requested = False
        last_capture_error = None
        while simulation_app.is_running():
            world.step(render=True)
            _, _, stop_requested = snapshots.state()
            if stop_requested:
                world.pause()
                break
            ticks += 1
            if ticks % args.capture_every:
                continue
            try:
                snapshots.publish(adapter.capture(sequence))
                sequence += 1
                last_capture_error = None
            except RuntimeError as error:
                # Camera annotators commonly need several rendered frames to warm up.
                snapshots.fail(str(error))
            except Exception as error:  # noqa: BLE001 - keep the gateway alive for diagnosis
                message = f"{type(error).__name__}: {error}"
                snapshots.fail(message)
                if message != last_capture_error:
                    print(f"Observation capture failed: {message}", file=sys.stderr, flush=True)
                    traceback.print_exc()
                    last_capture_error = message

        if not stop_requested:
            print(
                "Isaac application loop ended unexpectedly "
                f"after {ticks} ticks (exiting={simulation_app.is_exiting()}).",
                flush=True,
            )

        snapshots.disconnect()
        return 0
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=5)
        if snapshots is not None:
            snapshots.disconnect()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
