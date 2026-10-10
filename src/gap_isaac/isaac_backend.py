"""Read-only Isaac backend with bounded memory and hard motion interlocks."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

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
    StopRequest,
)


@dataclass(frozen=True)
class IsaacSnapshot:
    sequence: int
    captured_at: datetime
    world_frame: str
    rgb: bytes
    depth: bytes
    camera: CameraIntrinsics
    camera_pose: Pose
    end_effector_pose: Pose
    joints: JointState


class SnapshotBuffer:
    """Thread-safe handoff from Isaac's main thread to request threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: IsaacSnapshot | None = None
        self._connected = False
        self._error: str | None = None
        self._stop_requested = False

    def publish(self, snapshot: IsaacSnapshot) -> None:
        with self._lock:
            if self._snapshot and snapshot.sequence <= self._snapshot.sequence:
                raise ValueError("snapshot sequence must increase")
            self._snapshot = snapshot
            self._connected = True
            self._error = None

    def fail(self, error: str) -> None:
        with self._lock:
            self._error = error

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False

    def latest(self) -> IsaacSnapshot:
        with self._lock:
            if self._snapshot is None:
                raise RuntimeError("Isaac has not published an observation")
            return self._snapshot

    def state(self) -> tuple[bool, str | None, bool]:
        with self._lock:
            return self._connected, self._error, self._stop_requested

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True


class IsaacReadOnlyBackend:
    """Connector backend for the observation-only feasibility gate.

    Motion is deliberately impossible in this class. Enabling robot motion will
    require a separate backend and a separate review.
    """

    def __init__(
        self,
        snapshots: SnapshotBuffer,
        base_url: str = "http://127.0.0.1:8765",
        retained_frames: int = 2,
    ) -> None:
        if retained_frames < 1 or retained_frames > 8:
            raise ValueError("retained_frames must be between 1 and 8")
        self._snapshots = snapshots
        self._base_url = base_url.rstrip("/")
        self._retained_frames = retained_frames
        self._assets: OrderedDict[tuple[int, str], AssetPayload] = OrderedDict()
        self._asset_lock = threading.Lock()

    def health(self) -> HealthResponse:
        connected, error, stop_requested = self._snapshots.state()
        if stop_requested:
            status = "stopped"
            detail = "stop requested"
        elif error:
            status = "degraded"
            detail = error
        elif connected:
            status = "ready"
            detail = "read-only Isaac connector"
        else:
            status = "degraded"
            detail = "waiting for Isaac observation"
        return HealthResponse(
            status=status,
            simulator_connected=connected,
            motion_enabled=False,
            detail=detail,
        )

    def observation(self) -> ObservationDescriptor:
        snapshot = self._snapshots.latest()
        self._remember(snapshot.sequence, "rgb", snapshot.rgb)
        self._remember(snapshot.sequence, "depth", snapshot.depth)
        return ObservationDescriptor(
            sequence=snapshot.sequence,
            captured_at=snapshot.captured_at,
            world_frame=snapshot.world_frame,
            rgb_url=f"{self._base_url}/v1/assets/{snapshot.sequence}/rgb",
            rgb_sha256=hashlib.sha256(snapshot.rgb).hexdigest(),
            depth_url=f"{self._base_url}/v1/assets/{snapshot.sequence}/depth",
            depth_sha256=hashlib.sha256(snapshot.depth).hexdigest(),
            depth_encoding="32FC1",
            camera=snapshot.camera,
            camera_pose=snapshot.camera_pose,
            end_effector_pose=snapshot.end_effector_pose,
            joints=snapshot.joints,
        )

    def _remember(self, sequence: int, kind: str, body: bytes) -> None:
        key = (sequence, kind)
        with self._asset_lock:
            self._assets[key] = AssetPayload(body)
            self._assets.move_to_end(key)
            while len(self._assets) > self._retained_frames * 2:
                self._assets.popitem(last=False)

    def asset(self, sequence: int, kind: str) -> AssetPayload:
        if kind not in {"rgb", "depth"}:
            raise ValueError(f"unknown asset kind: {kind}")
        with self._asset_lock:
            return self._assets[(sequence, kind)]

    def execute_trajectory(self, request: ExecuteTrajectoryRequest) -> CommandResult:
        return CommandResult(
            command_id=request.command_id,
            status="failed",
            detail="motion is disabled by the read-only backend",
        )

    def control_gripper(self, request: GripperRequest) -> CommandResult:
        return CommandResult(
            command_id=request.command_id,
            status="failed",
            detail="motion is disabled by the read-only backend",
        )

    def stop(self, request: StopRequest) -> CommandResult:
        self._snapshots.request_stop()
        return CommandResult(
            command_id=request.command_id,
            status="succeeded",
            detail="stop latched; simulator loop will pause on its next tick",
        )
