"""Non-mutating end-to-end checks for the read-only Isaac connector."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

import httpx

from gap_isaac.protocol import ObservationDescriptor


@dataclass(frozen=True)
class ConnectorSmokeResult:
    sequence: int
    rgb_bytes: int
    depth_bytes: int
    joint_count: int
    trajectory_status: int
    gripper_status: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _same_origin(base_url: str, asset_url: str) -> bool:
    base = urlparse(base_url)
    asset = urlparse(asset_url)
    return (asset.scheme, asset.hostname, asset.port) == (base.scheme, base.hostname, base.port)


def _finite_observation(observation: ObservationDescriptor) -> None:
    values = [
        observation.camera.fx,
        observation.camera.fy,
        observation.camera.cx,
        observation.camera.cy,
        observation.camera_pose.position.x,
        observation.camera_pose.position.y,
        observation.camera_pose.position.z,
        observation.camera_pose.orientation.x,
        observation.camera_pose.orientation.y,
        observation.camera_pose.orientation.z,
        observation.camera_pose.orientation.w,
        observation.end_effector_pose.position.x,
        observation.end_effector_pose.position.y,
        observation.end_effector_pose.position.z,
        observation.end_effector_pose.orientation.x,
        observation.end_effector_pose.orientation.y,
        observation.end_effector_pose.orientation.z,
        observation.end_effector_pose.orientation.w,
        *observation.joints.positions_rad,
        *observation.joints.velocities_rad_s,
    ]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("observation contains non-finite numeric values")


def _validate_asset(body: bytes, expected_bytes: int, expected_sha256: str, name: str) -> None:
    if len(body) != expected_bytes:
        raise RuntimeError(f"{name} byte count {len(body)} does not match {expected_bytes}")
    actual = hashlib.sha256(body).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"{name} SHA-256 mismatch")


def run_connector_smoke(
    base_url: str,
    token: str,
    timeout_s: float = 30.0,
) -> ConnectorSmokeResult:
    """Validate observations and confirm both motion routes are interlocked."""

    if not token:
        raise ValueError("connector token must not be empty")
    normalized_base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=normalized_base, headers=headers, timeout=timeout_s) as client:
        health_response = client.get("/v1/health")
        health_response.raise_for_status()
        health = health_response.json()
        if health.get("status") != "ready" or not health.get("simulator_connected"):
            raise RuntimeError(f"connector is not ready: {health}")
        if health.get("motion_enabled") is not False:
            raise RuntimeError("connector unexpectedly reports motion enabled")

        observation_response = client.get("/v1/observations/latest")
        observation_response.raise_for_status()
        observation = ObservationDescriptor.model_validate(observation_response.json())
        _finite_observation(observation)
        if not _same_origin(normalized_base, observation.rgb_url) or not _same_origin(
            normalized_base, observation.depth_url
        ):
            raise RuntimeError("observation asset URL does not match connector origin")

        rgb_response = client.get(observation.rgb_url)
        rgb_response.raise_for_status()
        depth_response = client.get(observation.depth_url)
        depth_response.raise_for_status()
        rgb_bytes = observation.camera.width * observation.camera.height * 3
        depth_pixel_bytes = 4 if observation.depth_encoding == "32FC1" else 2
        depth_bytes = observation.camera.width * observation.camera.height * depth_pixel_bytes
        _validate_asset(rgb_response.content, rgb_bytes, observation.rgb_sha256, "RGB")
        _validate_asset(depth_response.content, depth_bytes, observation.depth_sha256, "depth")

        if not observation.joints.names:
            raise RuntimeError("observation does not contain robot joints")
        joint_name = observation.joints.names[0]
        joint_position = observation.joints.positions_rad[0]
        trajectory_response = client.post(
            "/v1/commands/trajectory",
            json={
                "protocol_version": "1.0",
                "command_id": "smoke-motion-must-be-disabled",
                "joint_names": [joint_name],
                "points": [{"time_from_start_s": 1.0, "positions_rad": [joint_position]}],
                "timeout_s": 2.0,
            },
        )
        gripper_response = client.post(
            "/v1/commands/gripper",
            json={
                "protocol_version": "1.0",
                "command_id": "smoke-gripper-must-be-disabled",
                "action": "open",
            },
        )
        if trajectory_response.status_code != 409 or gripper_response.status_code != 409:
            statuses = (
                f"trajectory={trajectory_response.status_code}, "
                f"gripper={gripper_response.status_code}"
            )
            raise RuntimeError(f"motion interlock failed: {statuses}")

    return ConnectorSmokeResult(
        sequence=observation.sequence,
        rgb_bytes=rgb_bytes,
        depth_bytes=depth_bytes,
        joint_count=len(observation.joints.names),
        trajectory_status=trajectory_response.status_code,
        gripper_status=gripper_response.status_code,
    )
