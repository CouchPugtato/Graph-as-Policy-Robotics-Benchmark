"""In-memory backend for connector development without Isaac Sim."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from gap_isaac.gateway import AssetPayload
from gap_isaac.protocol import (
    CameraIntrinsics,
    CommandResult,
    ExecuteTrajectoryRequest,
    GripperRequest,
    HealthResponse,
    JointState,
    ObservationDescriptor,
    Pose,
    Quaternion,
    StopRequest,
    Vector3,
)


def _identity_pose(frame_id: str = "world") -> Pose:
    return Pose(
        position=Vector3(x=0, y=0, z=0),
        orientation=Quaternion(x=0, y=0, z=0, w=1),
        frame_id=frame_id,
    )


class FakeBackend:
    def __init__(self, motion_enabled: bool = False) -> None:
        self.motion_enabled = motion_enabled
        self.stopped = False
        self.commands: list[str] = []
        self.assets = {"rgb": b"fake-rgb", "depth": b"fake-depth"}

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="stopped" if self.stopped else "ready",
            simulator_connected=True,
            motion_enabled=self.motion_enabled and not self.stopped,
            detail="fake backend",
        )

    def observation(self) -> ObservationDescriptor:
        return ObservationDescriptor(
            sequence=0,
            captured_at=datetime.now(timezone.utc),
            world_frame="world",
            rgb_url="http://127.0.0.1:8765/v1/assets/0/rgb",
            rgb_sha256=hashlib.sha256(self.assets["rgb"]).hexdigest(),
            depth_url="http://127.0.0.1:8765/v1/assets/0/depth",
            depth_sha256=hashlib.sha256(self.assets["depth"]).hexdigest(),
            depth_encoding="32FC1",
            camera=CameraIntrinsics(width=640, height=480, fx=500, fy=500, cx=320, cy=240),
            camera_pose=_identity_pose(),
            end_effector_pose=_identity_pose(),
            joints=JointState(names=("panda_joint1",), positions_rad=(0.0,)),
        )

    def asset(self, sequence: int, kind: str) -> AssetPayload:
        if sequence != 0 or kind not in self.assets:
            raise KeyError((sequence, kind))
        return AssetPayload(self.assets[kind])

    def execute_trajectory(self, request: ExecuteTrajectoryRequest) -> CommandResult:
        self.commands.append(request.command_id)
        return CommandResult(command_id=request.command_id, status="succeeded")

    def control_gripper(self, request: GripperRequest) -> CommandResult:
        self.commands.append(request.command_id)
        return CommandResult(command_id=request.command_id, status="succeeded")

    def stop(self, request: StopRequest) -> CommandResult:
        self.commands.append(request.command_id)
        self.stopped = True
        return CommandResult(command_id=request.command_id, status="succeeded")
