"""Shared, low-resolution pinhole lens settings; no simulator imports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

CAPTURE_RESOLUTION = (640, 480)
DEFAULT_HORIZONTAL_FOV_DEG = 65.0
# Wider full-scene view, angled to expose the basket interior and object sides.
GROCERY_CAMERA_EYE_M = (2.20, -2.20, 2.00)
GROCERY_CAMERA_TARGET_M = (0.30, 0.0, 0.65)


@dataclass(frozen=True)
class PinholeLens:
    width: int
    height: int
    horizontal_fov_deg: float
    horizontal_aperture: float
    vertical_aperture: float
    focal_length: float

    @property
    def focal_pixels(self) -> float:
        return self.width * self.focal_length / self.horizontal_aperture


def build_pinhole_lens(horizontal_fov_deg: float = DEFAULT_HORIZONTAL_FOV_DEG) -> PinholeLens:
    """Use USD camera units consistently; only aperture/focal ratios matter.

    Fix the aperture aspect ratio to the image ratio so pixels stay square.
    Limit view changes to a useful 30–90 degree range, without changing pixels.
    """
    if not math.isfinite(horizontal_fov_deg) or not 30.0 <= horizontal_fov_deg <= 90.0:
        raise ValueError("horizontal field of view must be finite and between 30 and 90 degrees")
    width, height = CAPTURE_RESOLUTION
    aperture = 20.955
    return PinholeLens(
        width=width,
        height=height,
        horizontal_fov_deg=horizontal_fov_deg,
        horizontal_aperture=aperture,
        vertical_aperture=aperture * height / width,
        focal_length=aperture / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0)),
    )


def apply_pinhole_lens(usd_camera: Any, lens: PinholeLens) -> None:
    """Author lens attributes only, in the caller's selected USD edit layer.

    Camera transforms, physics, stage files, clipping range and resolution
    are untouched. The connector uses a session layer; builders use root.
    """
    if not usd_camera:
        raise ValueError("capture camera prim is missing or is not a USD Camera")
    attributes = (
        (usd_camera.CreateProjectionAttr(), "perspective"),
        (usd_camera.CreateHorizontalApertureAttr(), lens.horizontal_aperture),
        (usd_camera.CreateVerticalApertureAttr(), lens.vertical_aperture),
        (usd_camera.CreateFocalLengthAttr(), lens.focal_length),
        (usd_camera.CreateHorizontalApertureOffsetAttr(), 0.0),
        (usd_camera.CreateVerticalApertureOffsetAttr(), 0.0),
    )
    for attribute, value in attributes:
        if not attribute.Set(value):
            raise RuntimeError("could not author capture camera lens")
