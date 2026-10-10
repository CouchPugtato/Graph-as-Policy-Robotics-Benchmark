"""Bounded, verified RGB-D captures for offline open-skill perception.

No Isaac, GPU, image-library, or model imports are needed to capture/inspect.
The optional CameraFrame conversion imports numpy only when explicitly called.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from gap_isaac.connector_smoke import _finite_observation, _same_origin, _validate_asset
from gap_isaac.protocol import HealthResponse, ObservationDescriptor

MAX_PIXELS = 640 * 480
DEPTH_MIN_M = 0.015
DEPTH_MAX_M = 20.0


def _sizes(observation: ObservationDescriptor) -> tuple[int, int]:
    # keep capture memory bounded before downloading the image assets
    pixels = observation.camera.width * observation.camera.height
    if pixels > MAX_PIXELS:
        raise ValueError("capture exceeds the 640x480 pixel budget")
    _finite_observation(observation)
    pose = observation.camera_pose
    if pose.frame_id != observation.world_frame:
        raise ValueError("camera pose must be expressed in the observation world frame")
    q = pose.orientation
    if not math.isclose(sum(v * v for v in (q.x, q.y, q.z, q.w)), 1.0, abs_tol=1e-4):
        raise ValueError("camera orientation must be a unit quaternion")
    return pixels * 3, pixels * (4 if observation.depth_encoding == "32FC1" else 2)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def rgb_png(rgb: bytes, width: int, height: int) -> bytes:
    """Encode the small RGB8 preview without an additional image dependency."""
    if width < 1 or height < 1 or width * height > MAX_PIXELS:
        raise ValueError("invalid or over-budget RGB dimensions")
    if len(rgb) != width * height * 3:
        raise ValueError("RGB byte count does not match dimensions")
    stride = width * 3
    rows = b"".join(b"\0" + rgb[y * stride : (y + 1) * stride] for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )


@dataclass(frozen=True)
class ObservationBundle:
    observation: ObservationDescriptor
    rgb: bytes
    depth: bytes

    def validate(self) -> None:
        rgb_size, depth_size = _sizes(self.observation)
        _validate_asset(self.rgb, rgb_size, self.observation.rgb_sha256, "RGB")
        _validate_asset(self.depth, depth_size, self.observation.depth_sha256, "depth")

    def depth_summary(self) -> dict[str, float | int | None]:
        self.validate()
        float_depth = self.observation.depth_encoding == "32FC1"
        fmt, scale = ("<f", 1.0) if float_depth else ("<H", 0.001)
        count, low, high = 0, math.inf, -math.inf
        for (raw,) in struct.iter_unpack(fmt, self.depth):
            value = raw * scale
            if math.isfinite(value) and DEPTH_MIN_M <= value <= DEPTH_MAX_M:
                count += 1
                low, high = min(low, value), max(high, value)
        pixels = self.observation.camera.width * self.observation.camera.height
        return {
            "valid_pixels": count,
            "total_pixels": pixels,
            "valid_fraction": count / pixels,
            "minimum_m": low if count else None,
            "maximum_m": high if count else None,
        }

    def pixel_to_world(self, x: int, y: int) -> tuple[float, float, float]:
        """Deproject image-plane depth through the ROS optical camera pose."""
        self.validate()
        camera = self.observation.camera
        if not (0 <= x < camera.width and 0 <= y < camera.height):
            raise ValueError("pixel is outside the image")
        if self.observation.depth_encoding == "32FC1":
            depth = struct.unpack_from("<f", self.depth, (y * camera.width + x) * 4)[0]
        else:
            depth = struct.unpack_from("<H", self.depth, (y * camera.width + x) * 2)[0] * 0.001
        if not math.isfinite(depth) or not DEPTH_MIN_M <= depth <= DEPTH_MAX_M:
            raise ValueError("pixel has invalid metric depth")
        point = ((x - camera.cx) * depth / camera.fx, (y - camera.cy) * depth / camera.fy, depth)
        pose = self.observation.camera_pose
        q = pose.orientation

        # Rotation via v' = v + 2*w*(q_xyz cross v) + 2*(q_xyz cross (q_xyz cross v)).
        def cross(a: tuple, b: tuple) -> tuple:
            return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

        first = cross((q.x, q.y, q.z), point)
        second = cross((q.x, q.y, q.z), first)
        origin = (pose.position.x, pose.position.y, pose.position.z)
        return tuple(origin[i] + point[i] + 2 * (q.w * first[i] + second[i]) for i in range(3))

    def to_gap_camera(self, name: str = "exterior") -> dict[str, Any]:
        """Convert to the real upstream CameraFrame shape (numpy optional)."""
        self.validate()
        import numpy as np

        camera = self.observation.camera
        shape = (camera.height, camera.width)
        dtype = "<f4" if self.observation.depth_encoding == "32FC1" else "<u2"
        depth = np.frombuffer(self.depth, dtype=dtype).reshape(shape).astype(np.float32)
        if dtype == "<u2":
            depth *= 0.001
        pose = self.observation.camera_pose
        q = pose.orientation
        return {
            "name": name,
            "rgb": np.frombuffer(self.rgb, dtype=np.uint8).reshape(*shape, 3).copy(),
            "depth": depth,
            "intrinsics": np.array(
                [[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]],
                dtype=np.float64,
            ),
            "pose": {
                "position": pose.position.model_dump(),
                "rotation": {"w": q.w, "x": q.x, "y": q.y, "z": q.z},
            },
        }

    def save(self, directory: Path) -> None:
        """Create a new bundle, never overwrite an earlier experiment capture."""
        self.validate()
        if not self.depth_summary()["valid_pixels"]:
            raise ValueError("capture has no valid metric depth pixels")
        preview = rgb_png(self.rgb, self.observation.camera.width, self.observation.camera.height)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "rgb.bin").write_bytes(self.rgb)
        (directory / "depth.bin").write_bytes(self.depth)
        (directory / "rgb.png").write_bytes(preview)
        manifest = {
            "schema_version": 1,
            "source": "connector-rgbd-only",
            "camera_axes": "ros-optical-x-right-y-down-z-forward",
            "depth_semantics": "distance-to-image-plane",
            "byte_order": "little-endian",
            "preview_sha256": hashlib.sha256(preview).hexdigest(),
            "observation": self.observation.model_dump(mode="json"),
        }
        (directory / "observation.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> ObservationBundle:
        manifest_path = directory / "observation.json"
        if manifest_path.stat().st_size > 64 * 1024:
            raise ValueError("observation manifest exceeds the size budget")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "source": "connector-rgbd-only",
            "camera_axes": "ros-optical-x-right-y-down-z-forward",
            "depth_semantics": "distance-to-image-plane",
            "byte_order": "little-endian",
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("unsupported observation bundle contract")
        observation = ObservationDescriptor.model_validate(manifest["observation"])
        rgb_size, depth_size = _sizes(observation)
        for filename, size in (("rgb.bin", rgb_size), ("depth.bin", depth_size)):
            if (directory / filename).stat().st_size != size:
                raise ValueError(f"{filename} byte count does not match descriptor")
        bundle = cls(
            observation,
            (directory / "rgb.bin").read_bytes(),
            (directory / "depth.bin").read_bytes(),
        )
        bundle.validate()
        preview_path = directory / "rgb.png"
        if preview_path.stat().st_size > MAX_PIXELS * 4 + 4096:
            raise ValueError("RGB preview exceeds the size budget")
        preview = preview_path.read_bytes()
        if hashlib.sha256(preview).hexdigest() != manifest.get("preview_sha256"):
            raise ValueError("RGB preview SHA-256 mismatch")
        if preview != rgb_png(bundle.rgb, observation.camera.width, observation.camera.height):
            raise ValueError("RGB preview does not match the captured pixels")
        return bundle


def _bounded_get(client: httpx.Client, url: str, limit: int) -> bytes:
    with client.stream("GET", url) as response:
        response.raise_for_status()
        body = bytearray()
        for block in response.iter_bytes(chunk_size=16 * 1024):
            if len(body) + len(block) > limit:
                raise ValueError("connector response exceeds capture byte budget")
            body.extend(block)
        return bytes(body)


def capture_observation(base_url: str, token: str) -> ObservationBundle:
    """GET-only capture; retry an evicted sequence at most three times."""
    if not token:
        raise ValueError("connector token must not be empty")
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
        raise ValueError("connector URL must be HTTP(S) without embedded credentials")
    with httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
        follow_redirects=False,
    ) as client:
        health = HealthResponse.model_validate_json(_bounded_get(client, "/v1/health", 4096))
        if health.motion_enabled or health.status != "ready" or not health.simulator_connected:
            raise ValueError("capture requires a ready read-only connector")
        for attempt in range(3):
            observation = ObservationDescriptor.model_validate_json(
                _bounded_get(client, "/v1/observations/latest", 64 * 1024)
            )
            sizes = _sizes(observation)
            urls = (observation.rgb_url, observation.depth_url)
            if not all(_same_origin(base_url, url) for url in urls):
                raise ValueError("asset origin mismatch; refusing to forward connector credential")
            for kind, url in zip(("rgb", "depth"), urls, strict=True):
                asset = urlparse(url)
                if (
                    asset.username
                    or asset.password
                    or asset.query
                    or asset.fragment
                    or (asset.path != f"/v1/assets/{observation.sequence}/{kind}")
                ):
                    raise ValueError("asset URL does not match the captured sequence")
            try:
                rgb, depth = (
                    _bounded_get(client, url, size) for url, size in zip(urls, sizes, strict=True)
                )
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 404 and attempt < 2:
                    continue
                raise
            bundle = ObservationBundle(observation, rgb, depth)
            bundle.validate()
            return bundle
    raise RuntimeError("no coherent frame could be captured")
