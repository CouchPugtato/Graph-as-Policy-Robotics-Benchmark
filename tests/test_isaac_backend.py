from datetime import datetime, timezone

import pytest

from gap_isaac.isaac_backend import IsaacReadOnlyBackend, IsaacSnapshot, SnapshotBuffer
from gap_isaac.protocol import (
    CameraIntrinsics,
    ExecuteTrajectoryRequest,
    JointState,
    Pose,
    Quaternion,
    TrajectoryPoint,
    Vector3,
)


def pose() -> Pose:
    return Pose(
        position=Vector3(x=0, y=0, z=1),
        orientation=Quaternion(x=0, y=0, z=0, w=1),
        frame_id="world",
    )


def snapshot(sequence: int) -> IsaacSnapshot:
    return IsaacSnapshot(
        sequence=sequence,
        captured_at=datetime.now(timezone.utc),
        world_frame="world",
        rgb=f"rgb-{sequence}".encode(),
        depth=f"depth-{sequence}".encode(),
        camera=CameraIntrinsics(width=2, height=2, fx=1, fy=1, cx=1, cy=1),
        camera_pose=pose(),
        end_effector_pose=pose(),
        joints=JointState(names=("joint",), positions_rad=(0.0,)),
    )


def test_backend_waits_for_first_frame_and_never_enables_motion() -> None:
    buffer = SnapshotBuffer()
    backend = IsaacReadOnlyBackend(buffer)
    health = backend.health()
    assert health.status == "degraded"
    assert health.motion_enabled is False

    with pytest.raises(RuntimeError, match="has not published"):
        backend.observation()

    buffer.publish(snapshot(1))
    observation = backend.observation()
    assert backend.health().status == "ready"
    assert backend.asset(1, "rgb").body == b"rgb-1"
    assert observation.rgb_url.endswith("/v1/assets/1/rgb")


def test_backend_bounds_assets_and_rejects_motion_in_depth() -> None:
    buffer = SnapshotBuffer()
    backend = IsaacReadOnlyBackend(buffer, retained_frames=1)
    buffer.publish(snapshot(1))
    backend.observation()
    buffer.publish(snapshot(2))
    backend.observation()

    with pytest.raises(KeyError):
        backend.asset(1, "rgb")
    assert backend.asset(2, "depth").body == b"depth-2"

    request = ExecuteTrajectoryRequest(
        command_id="must-not-move",
        joint_names=("joint",),
        points=(TrajectoryPoint(time_from_start_s=1, positions_rad=(0.1,)),),
        timeout_s=2,
    )
    assert backend.execute_trajectory(request).status == "failed"


def test_snapshot_sequence_must_increase() -> None:
    buffer = SnapshotBuffer()
    buffer.publish(snapshot(2))
    with pytest.raises(ValueError, match="must increase"):
        buffer.publish(snapshot(2))
