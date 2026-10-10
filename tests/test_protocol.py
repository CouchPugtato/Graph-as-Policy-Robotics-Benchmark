from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from gap_isaac.protocol import (
    CameraIntrinsics,
    ExecuteTrajectoryRequest,
    JointState,
    ObservationDescriptor,
    Pose,
    Quaternion,
    TrajectoryPoint,
    Vector3,
)


def pose() -> Pose:
    return Pose(
        position=Vector3(x=0, y=0, z=0),
        orientation=Quaternion(x=0, y=0, z=0, w=1),
        frame_id="world",
    )


def test_observation_round_trip() -> None:
    observation = ObservationDescriptor(
        sequence=1,
        captured_at=datetime.now(timezone.utc),
        world_frame="world",
        rgb_url="http://127.0.0.1:8765/assets/1/rgb",
        rgb_sha256="a" * 64,
        rgb_encoding="RGB8",
        depth_url="http://127.0.0.1:8765/assets/1/depth",
        depth_sha256="b" * 64,
        depth_encoding="32FC1",
        camera=CameraIntrinsics(width=640, height=480, fx=500, fy=500, cx=320, cy=240),
        camera_pose=pose(),
        end_effector_pose=pose(),
        joints=JointState(names=("joint1",), positions_rad=(0.0,)),
    )
    assert ObservationDescriptor.model_validate_json(observation.model_dump_json()) == observation


def test_trajectory_requires_monotonic_complete_points() -> None:
    with pytest.raises(ValidationError):
        ExecuteTrajectoryRequest(
            command_id="test",
            joint_names=("joint1", "joint2"),
            points=(
                TrajectoryPoint(time_from_start_s=1, positions_rad=(0.0, 0.0)),
                TrajectoryPoint(time_from_start_s=0.5, positions_rad=(0.1, 0.1)),
            ),
            timeout_s=10,
        )
