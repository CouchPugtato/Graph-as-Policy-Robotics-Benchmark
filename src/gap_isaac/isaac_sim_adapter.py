"""Small adapter for collecting Isaac Sim state on its main thread.

This module intentionally has no Isaac imports. The standalone launcher creates
Isaac handles after SimulationApp starts and passes them into this adapter.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from gap_isaac.isaac_backend import IsaacSnapshot
from gap_isaac.protocol import CameraIntrinsics, JointState, Pose, Quaternion, Vector3


def _pose(position: Any, orientation_wxyz: Any, frame_id: str = "world") -> Pose:
    return Pose(
        position=Vector3(x=float(position[0]), y=float(position[1]), z=float(position[2])),
        orientation=Quaternion(
            x=float(orientation_wxyz[1]),
            y=float(orientation_wxyz[2]),
            z=float(orientation_wxyz[3]),
            w=float(orientation_wxyz[0]),
        ),
        frame_id=frame_id,
    )


def _dof_names(robot: Any, expected_count: int) -> tuple[str, ...]:
    """Return ordered DOF names across supported Isaac articulation wrappers."""

    names = getattr(robot, "dof_names", None)
    if names is None:
        names = getattr(robot, "joint_names", None)
    if names is None:
        raise RuntimeError("robot articulation does not expose DOF names")
    result = tuple(str(name) for name in names)
    if len(result) != expected_count:
        raise RuntimeError(
            f"DOF name count {len(result)} does not match joint state count {expected_count}"
        )
    return result


class IsaacSimObservationAdapter:
    def __init__(self, camera: Any, robot: Any, end_effector: Any) -> None:
        self._camera = camera
        self._robot = robot
        self._end_effector = end_effector

    def capture(self, sequence: int) -> IsaacSnapshot:
        rgb = self._camera.get_rgb()
        depth = self._camera.get_depth()
        if rgb is None or depth is None:
            raise RuntimeError("camera frame is not ready")

        intrinsics = self._camera.get_intrinsics_matrix()
        camera_position, camera_orientation = self._camera.get_world_pose(camera_axes="ros")
        ee_position, ee_orientation = self._end_effector.get_world_pose()
        joint_state = self._robot.get_joints_state()
        if joint_state is None:
            raise RuntimeError("robot joint state is not ready")

        positions = tuple(float(value) for value in joint_state.positions)
        velocities = (
            tuple(float(value) for value in joint_state.velocities)
            if joint_state.velocities is not None
            else ()
        )

        rgb_bytes = rgb.astype("uint8", copy=False).tobytes(order="C")
        depth_bytes = depth.astype("<f4", copy=False).tobytes(order="C")
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        return IsaacSnapshot(
            sequence=sequence,
            captured_at=datetime.now(timezone.utc),
            world_frame="world",
            rgb=rgb_bytes,
            depth=depth_bytes,
            camera=CameraIntrinsics(
                width=width,
                height=height,
                fx=float(intrinsics[0, 0]),
                fy=float(intrinsics[1, 1]),
                cx=float(intrinsics[0, 2]),
                cy=float(intrinsics[1, 2]),
            ),
            camera_pose=_pose(camera_position, camera_orientation),
            end_effector_pose=_pose(ee_position, ee_orientation),
            joints=JointState(
                names=_dof_names(self._robot, len(positions)),
                positions_rad=positions,
                velocities_rad_s=velocities,
            ),
        )
