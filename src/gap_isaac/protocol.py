"""Versioned messages for the Windows connector gateway.

These messages are the stable boundary between the native Windows Isaac process
and the policy runtime (currently native Windows).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOL_VERSION = "1.0"


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Vector3(ProtocolModel):
    x: float
    y: float
    z: float


class Quaternion(ProtocolModel):
    x: float
    y: float
    z: float
    w: float


class Pose(ProtocolModel):
    position: Vector3
    orientation: Quaternion
    frame_id: str


class CameraIntrinsics(ProtocolModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float


class JointState(ProtocolModel):
    names: tuple[str, ...]
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...] = ()

    @model_validator(mode="after")
    def matching_lengths(self) -> JointState:
        if len(self.names) != len(self.positions_rad):
            raise ValueError("joint names and positions must have equal length")
        if self.velocities_rad_s and len(self.names) != len(self.velocities_rad_s):
            raise ValueError("joint velocities must be empty or match joint names")
        return self


class ObservationDescriptor(ProtocolModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    sequence: int = Field(ge=0)
    captured_at: datetime
    world_frame: str
    rgb_url: str
    rgb_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rgb_encoding: Literal["RGB8"] = "RGB8"
    depth_url: str
    depth_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    depth_encoding: Literal["32FC1", "16UC1"]
    camera: CameraIntrinsics
    camera_pose: Pose
    end_effector_pose: Pose
    joints: JointState


class TrajectoryPoint(ProtocolModel):
    time_from_start_s: float = Field(ge=0)
    positions_rad: tuple[float, ...]


class ExecuteTrajectoryRequest(ProtocolModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    command_id: str = Field(min_length=1, max_length=128)
    joint_names: tuple[str, ...] = Field(min_length=1)
    points: tuple[TrajectoryPoint, ...] = Field(min_length=1)
    timeout_s: float = Field(gt=0, le=300)

    @model_validator(mode="after")
    def valid_trajectory(self) -> ExecuteTrajectoryRequest:
        expected = len(self.joint_names)
        previous = -1.0
        for point in self.points:
            if len(point.positions_rad) != expected:
                raise ValueError("every trajectory point must contain all joints")
            if point.time_from_start_s <= previous:
                raise ValueError("trajectory times must be strictly increasing")
            previous = point.time_from_start_s
        return self


class CommandResult(ProtocolModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    command_id: str
    status: Literal["accepted", "succeeded", "failed", "canceled", "timed_out"]
    detail: str = ""


class StopRequest(ProtocolModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    command_id: str = Field(min_length=1, max_length=128)


class GripperRequest(ProtocolModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    command_id: str = Field(min_length=1, max_length=128)
    action: Literal["open", "close"]
    width_m: float | None = Field(default=None, ge=0, le=0.2)
    force_n: float | None = Field(default=None, gt=0, le=200)


class HealthResponse(ProtocolModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    status: Literal["ready", "degraded", "stopped"]
    simulator_connected: bool
    motion_enabled: bool
    detail: str = ""
