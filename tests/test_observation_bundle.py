import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from gap_isaac import cli
from gap_isaac import observation_bundle as module
from gap_isaac.fake_backend import FakeBackend
from gap_isaac.observation_bundle import ObservationBundle, capture_observation, rgb_png
from gap_isaac.protocol import CameraIntrinsics, Quaternion


def make_bundle(encoding: str = "32FC1") -> ObservationBundle:
    rgb = bytes((255, 0, 0, 0, 255, 0))
    depth = struct.pack("<ff", 2.0, 0.0) if encoding == "32FC1" else struct.pack("<HH", 2000, 0)
    observation = (
        FakeBackend()
        .observation()
        .model_copy(
            update={
                "camera": CameraIntrinsics(width=2, height=1, fx=2, fy=2, cx=0, cy=0),
                "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
                "depth_sha256": hashlib.sha256(depth).hexdigest(),
                "depth_encoding": encoding,
            }
        )
    )
    return ObservationBundle(observation, rgb, depth)


def install_transport(monkeypatch, bundle, mutate=None):
    requests = []
    client_class = httpx.Client

    def handler(request):
        requests.append(request)
        if mutate:
            override = mutate(request)
            if override is not None:
                return override
        paths = {
            "/v1/health": httpx.Response(
                200,
                json={
                    "status": "ready",
                    "simulator_connected": True,
                    "motion_enabled": False,
                },
            ),
            "/v1/observations/latest": httpx.Response(
                200, json=bundle.observation.model_dump(mode="json")
            ),
            "/v1/assets/0/rgb": httpx.Response(200, content=bundle.rgb),
            "/v1/assets/0/depth": httpx.Response(200, content=bundle.depth),
        }
        return paths[request.url.path]

    monkeypatch.setattr(
        module.httpx,
        "Client",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(handler)),
    )
    return requests


@pytest.mark.parametrize("encoding", ["32FC1", "16UC1"])
def test_save_load_and_metric_depth(tmp_path: Path, encoding: str):
    bundle = make_bundle(encoding)
    directory = tmp_path / "frame"
    bundle.save(directory)
    loaded = ObservationBundle.load(directory)
    assert loaded == bundle
    assert loaded.depth_summary() == {
        "valid_pixels": 1,
        "total_pixels": 2,
        "valid_fraction": 0.5,
        "minimum_m": 2.0,
        "maximum_m": 2.0,
    }
    assert loaded.pixel_to_world(0, 0) == (0, 0, 2)
    manifest = (directory / "observation.json").read_text()
    assert "Authorization" not in manifest
    assert "token" not in manifest
    with pytest.raises(FileExistsError):
        bundle.save(directory)


def test_deprojection_rotates_optical_frame_to_world():
    bundle = make_bundle()
    pose = bundle.observation.camera_pose.model_copy(
        update={
            "orientation": Quaternion(x=0, y=math.sqrt(0.5), z=0, w=math.sqrt(0.5)),
        }
    )
    bundle = ObservationBundle(
        bundle.observation.model_copy(update={"camera_pose": pose}), bundle.rgb, bundle.depth
    )
    assert bundle.pixel_to_world(0, 0) == pytest.approx((2, 0, 0))


def test_png_is_real_rgb_not_raw_bytes():
    data = rgb_png(make_bundle().rgb, 2, 1)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, chunks = 8, {}
    while offset < len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + size]
        crc = struct.unpack_from(">I", data, offset + 8 + size)[0]
        assert crc == zlib.crc32(kind + body)
        chunks[kind] = body
        offset += size + 12
    assert struct.unpack(">IIBBBBB", chunks[b"IHDR"]) == (2, 1, 8, 2, 0, 0, 0)
    assert zlib.decompress(chunks[b"IDAT"]) == b"\0" + make_bundle().rgb


def test_capture_is_get_only_and_authenticated(monkeypatch):
    bundle = make_bundle()
    requests = install_transport(monkeypatch, bundle)
    assert capture_observation("http://127.0.0.1:8765", "private-token") == bundle
    assert len(requests) == 4
    assert all(r.method == "GET" for r in requests)
    assert all(r.headers["Authorization"] == "Bearer private-token" for r in requests)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1/assets/0/rgb",
        "http://127.0.0.1:8765/v1/assets/1/rgb",
        "http://127.0.0.1:8765/v1/assets/0/rgb?token=secret",
    ],
)
def test_asset_validation_precedes_credential_forwarding(monkeypatch, url):
    bundle = make_bundle()
    bundle = ObservationBundle(
        bundle.observation.model_copy(update={"rgb_url": url}), bundle.rgb, bundle.depth
    )
    requests = install_transport(monkeypatch, bundle)
    with pytest.raises(ValueError):
        capture_observation("http://127.0.0.1:8765", "private-token")
    assert len(requests) == 2


def test_motion_enabled_connector_is_rejected(monkeypatch):
    def mutate(request):
        if request.url.path == "/v1/health":
            return httpx.Response(
                200,
                json={
                    "status": "ready",
                    "simulator_connected": True,
                    "motion_enabled": True,
                },
            )

    requests = install_transport(monkeypatch, make_bundle(), mutate)
    with pytest.raises(ValueError, match="read-only"):
        capture_observation("http://127.0.0.1:8765", "token")
    assert len(requests) == 1


def test_evicted_frame_retries_are_bounded(monkeypatch):
    def mutate(request):
        if request.url.path.endswith("/depth"):
            return httpx.Response(404)

    requests = install_transport(monkeypatch, make_bundle(), mutate)
    with pytest.raises(httpx.HTTPStatusError):
        capture_observation("http://127.0.0.1:8765", "token")
    assert sum(r.url.path == "/v1/observations/latest" for r in requests) == 3


def test_oversize_stream_rejected(monkeypatch):
    def mutate(request):
        if request.url.path.endswith("/rgb"):
            return httpx.Response(200, content=b"x" * 100)

    install_transport(monkeypatch, make_bundle(), mutate)
    with pytest.raises(ValueError, match="byte budget"):
        capture_observation("http://127.0.0.1:8765", "token")


def test_corrupt_saved_frame_rejected(tmp_path):
    directory = tmp_path / "frame"
    make_bundle().save(directory)
    (directory / "rgb.bin").write_bytes(b"x" * 6)
    with pytest.raises(RuntimeError, match="SHA-256"):
        ObservationBundle.load(directory)


def test_corrupt_preview_rejected(tmp_path):
    directory = tmp_path / "frame"
    make_bundle().save(directory)
    (directory / "rgb.png").write_bytes(b"not a PNG")
    with pytest.raises(ValueError, match="preview SHA-256"):
        ObservationBundle.load(directory)


def test_large_capture_is_rejected_before_download(monkeypatch):
    bundle = make_bundle()
    camera = bundle.observation.camera.model_copy(update={"width": 1920, "height": 1080})
    bundle = ObservationBundle(
        bundle.observation.model_copy(update={"camera": camera}), bundle.rgb, bundle.depth
    )
    requests = install_transport(monkeypatch, bundle)
    with pytest.raises(ValueError, match="pixel budget"):
        capture_observation("http://127.0.0.1:8765", "token")
    assert len(requests) == 2


@pytest.mark.parametrize(
    "update, message",
    [
        ({"frame_id": "unknown"}, "world frame"),
        ({"orientation": Quaternion(x=0, y=0, z=0, w=2)}, "unit quaternion"),
    ],
)
def test_bad_camera_pose_rejected(update, message):
    bundle = make_bundle()
    pose = bundle.observation.camera_pose.model_copy(update=update)
    bundle = ObservationBundle(
        bundle.observation.model_copy(update={"camera_pose": pose}), bundle.rgb, bundle.depth
    )
    with pytest.raises(ValueError, match=message):
        bundle.validate()


def test_invalid_pixel_rejected():
    with pytest.raises(ValueError, match="invalid metric depth"):
        make_bundle().pixel_to_world(1, 0)
    with pytest.raises(ValueError, match="outside"):
        make_bundle().pixel_to_world(2, 0)


def test_no_valid_depth_does_not_create_artifacts(tmp_path):
    bundle = make_bundle()
    depth = b"\0" * len(bundle.depth)
    bundle = ObservationBundle(
        bundle.observation.model_copy(
            update={
                "depth_sha256": hashlib.sha256(depth).hexdigest(),
            }
        ),
        bundle.rgb,
        depth,
    )
    with pytest.raises(ValueError, match="no valid"):
        bundle.save(tmp_path / "frame")
    assert not (tmp_path / "frame").exists()


def test_capture_and_inspect_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("GAP_ISAAC_CONNECTOR_TOKEN", "private-token")
    install_transport(monkeypatch, make_bundle())
    runner = CliRunner()
    directory = tmp_path / "frame"
    result = runner.invoke(cli.app, ["capture-observation", "--output", str(directory)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["models_loaded"] is False
    assert "private-token" not in result.output
    result = runner.invoke(cli.app, ["inspect-observation", str(directory)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["depth"]["valid_pixels"] == 1
    result = runner.invoke(cli.app, ["capture-observation", "--output", str(directory)])
    assert result.exit_code == 1
    assert "already exists" in result.output
