import itertools
import math
from pathlib import Path

import pytest

from gap_isaac.camera_config import (
    CAPTURE_RESOLUTION,
    GROCERY_CAMERA_EYE_M,
    GROCERY_CAMERA_TARGET_M,
    apply_pinhole_lens,
    build_pinhole_lens,
)
from gap_isaac.config import load_task
from gap_isaac.grocery_scene import build_grocery_scene_plan
from isaac.run_read_only_connector import parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_wide_camera_keeps_pixel_budget_and_square_pixels():
    lens = build_pinhole_lens()
    assert (lens.width, lens.height) == CAPTURE_RESOLUTION == (640, 480)
    assert lens.horizontal_fov_deg == 65
    assert lens.focal_pixels == pytest.approx(502.2993846776)
    assert lens.height * lens.focal_length / lens.vertical_aperture == pytest.approx(
        lens.focal_pixels
    )
    assert math.degrees(2 * math.atan(lens.horizontal_aperture / (2 * lens.focal_length))) == (
        pytest.approx(65)
    )


@pytest.mark.parametrize("value", [0, 29, 91, float("nan"), float("inf")])
def test_invalid_fov_rejected(value):
    with pytest.raises(ValueError, match="between 30 and 90"):
        build_pinhole_lens(value)


def test_only_lens_attributes_are_authored():
    values = {}

    class Attribute:
        def __init__(self, name):
            self.name = name

        def Set(self, value):
            values[self.name] = value
            return True

    class Camera:
        def __getattr__(self, name):
            return lambda: Attribute(name)

    lens = build_pinhole_lens()
    apply_pinhole_lens(Camera(), lens)
    assert values == {
        "CreateProjectionAttr": "perspective",
        "CreateHorizontalApertureAttr": lens.horizontal_aperture,
        "CreateVerticalApertureAttr": lens.vertical_aperture,
        "CreateFocalLengthAttr": lens.focal_length,
        "CreateHorizontalApertureOffsetAttr": 0.0,
        "CreateVerticalApertureOffsetAttr": 0.0,
    }
    with pytest.raises(ValueError, match="missing"):
        apply_pinhole_lens(None, lens)


def project_to_image(point):
    eye = GROCERY_CAMERA_EYE_M
    delta = tuple(GROCERY_CAMERA_TARGET_M[i] - eye[i] for i in range(3))
    distance = math.sqrt(sum(v * v for v in delta))
    forward = tuple(v / distance for v in delta)
    horizontal = math.hypot(forward[0], forward[1])
    right = (forward[1] / horizontal, -forward[0] / horizontal, 0.0)
    down = (
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
    )
    focal = build_pinhole_lens().focal_pixels

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    vector = tuple(point[i] - eye[i] for i in range(3))
    optical_z = dot(vector, forward)
    assert optical_z > 0
    return (
        320 + focal * dot(vector, right) / optical_z,
        240 + focal * dot(vector, down) / optical_z,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_proxy_object_bounds_fit_reduced_demo_camera(seed):
    """Geometric coverage only: visibility/occlusion still needs a real capture."""
    plan = build_grocery_scene_plan(
        load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml"), seed, "alphabet_soup"
    )
    for item in plan.objects:
        yaw = math.radians(item.yaw_deg)
        width, depth, height = item.size_m
        for sx, sy, sz in itertools.product((-1, 1), repeat=3):
            x, y = sx * width / 2, sy * depth / 2
            z = (0 if sz < 0 else height + 0.020) if item.is_container else sz * height / 2
            point = (
                item.position_m[0] + x * math.cos(yaw) - y * math.sin(yaw),
                item.position_m[1] + x * math.sin(yaw) + y * math.cos(yaw),
                item.position_m[2] + z,
            )
            u, v = project_to_image(point)
            assert 8 < u < 632, (seed, item.object_id, u)
            assert 8 < v < 472, (seed, item.object_id, v)


def test_robot_framing_envelope_fits_with_margin():
    """Conservative framing envelope, not a measured robot mesh or occlusion test."""
    for point in itertools.product((-0.30, 1.05), (-0.60, 0.60), (0.0, 1.50)):
        u, v = project_to_image(point)
        assert 16 < u < 624, (point, u)
        assert 16 < v < 464, (point, v)


def test_connector_fov_option_has_wide_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["connector", "--stage", "scene.usd"])
    assert parse_args().horizontal_fov_deg == 65
    monkeypatch.setattr(
        "sys.argv", ["connector", "--stage", "scene.usd", "--horizontal-fov-deg", "70"]
    )
    assert parse_args().horizontal_fov_deg == 70
