"""CPU-only reconstruction of visible target surface geometry from saved RGB-D."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import types


def validate_cloud_summary(point_count: int, minimum: list, maximum: list):
    if point_count < 10:
        raise ValueError("At least ten valid masked depth points are required")
    if not all(math.isfinite(float(value)) for value in [*minimum, *maximum]):
        raise ValueError("Point cloud contains nonfinite coordinates")
    if minimum[2] < -0.02:
        raise ValueError("Target cloud falls below the ground; check depth/optical calibration")
    spans = [hi - lo for lo, hi in zip(minimum, maximum, strict=True)]
    if any(span < 0 or span > 0.5 for span in spans):
        raise ValueError("Target cloud exceeds the bounded half-meter geometry budget")


def load_geometry_tools(checkout: Path):
    # Mount the source layout at its declared package paths, without editing it
    # or installing the full graph runtime / simulator dependencies.
    for name, path in (("gap_skills", checkout), ("gap_skills.tools", checkout / "tools"),
                       ("gap_skills.tools.geometry", checkout / "tools/geometry")):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    name = "gap_skills.tools.geometry.tools"
    spec = importlib.util.spec_from_file_location(name, checkout / "tools/geometry/tools.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(root: Path, segmentation: Path) -> dict:
    from gap_isaac.observation_bundle import ObservationBundle
    from gap_isaac.perception_preflight import SKILLS_REVISION
    data = json.loads(segmentation.read_text(encoding="utf-8"))
    if not data.get("ok") or not data.get("masks"):
        raise ValueError("A successful saved SAM3 mask result is required")
    image = Path(data["image"])
    bundle = ObservationBundle.load(image.parent)
    # the mask and depth must belong to the same captured image
    if hashlib.sha256(image.read_bytes()).hexdigest() != data["image_sha256"]:
        raise ValueError("SAM3 image does not match the verified RGB-D capture")
    mask_path = segmentation.parent / data["masks"][0]["file"]
    if mask_path.resolve().parent != segmentation.parent.resolve():
        raise ValueError("Mask must belong to the saved segmentation result directory")
    checkout = root / "cache/upstream/open-robot-skills"
    for args, expected in ((["rev-parse", "HEAD"], SKILLS_REVISION),
                           (["status", "--porcelain"], "")):
        actual = subprocess.run(["git", "-C", str(checkout), *args],
                                capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        if actual != expected:
            raise ValueError("Geometry tools must be clean and pinned")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2",
                      OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    def blocked(*args, **kwargs):
        raise RuntimeError("Network disabled during CPU geometry reconstruction")
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    import numpy as np
    from PIL import Image
    with Image.open(mask_path) as mask_image:
        if mask_image.mode != "L" or mask_image.size != (640, 480):
            raise ValueError("Expected a 640x480 grayscale mask")
        mask = np.array(mask_image, dtype=np.uint8)
    if not np.isin(mask, (0, 255)).all():
        raise ValueError("Mask must contain only zero and 255")
    if int(np.count_nonzero(mask)) != data["masks"][0]["foreground_pixels"]:
        raise ValueError("Mask area differs from saved SAM3 result")
    camera = bundle.to_gap_camera()
    tools = load_geometry_tools(checkout)
    cloud = tools.mask_to_world_points(mask=mask, depth=camera["depth"],
                                      intrinsics=camera["intrinsics"], camera_pose=camera["pose"])["points"]
    points = cloud["points"]
    if len(points) < 10 or not np.isfinite(points).all():
        raise ValueError("Insufficient finite masked world points")
    minimum, maximum = points.min(axis=0).tolist(), points.max(axis=0).tolist()
    validate_cloud_summary(len(points), minimum, maximum)
    valid = (mask != 0) & np.isfinite(camera["depth"]) & (camera["depth"] >= .015) & (camera["depth"] <= 20)
    ys, xs = np.nonzero(valid)
    # Compare actual upstream output against an independent optical/quaternion
    # deprojection implementation at eight corresponding pixels.
    errors = []
    for index in np.linspace(0, len(points) - 1, min(8, len(points)), dtype=int):
        reference = bundle.pixel_to_world(int(xs[index]), int(ys[index]))
        errors.append(float(np.linalg.norm(points[index] - reference)))
    if max(errors) > 1e-5:
        raise ValueError("Upstream/world deprojection disagrees with reference")
    np.random.seed(0)  # Upstream OBB introduces 0.1 mm noise; make it reproducible.
    obb = tools.compute_obb(points=cloud)["obb"]
    dimensions = [2 * obb["extent"][axis] for axis in ("x", "y", "z")]
    if not all(math.isfinite(value) and 0 < value <= .5 for value in dimensions):
        raise ValueError("Invalid bounded OBB dimensions")
    output = root / "results/perception" / ("geometry-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "world_points.npy", points, allow_pickle=False)
    depth = camera["depth"][valid]
    report = {"ok": True, "geometry_source": "single-view-masked-rgbd-visible-surface",
              "target_id": data["target_id"], "selection_source": data["selection_source"],
              "segmentation": str(segmentation.resolve()), "bundle": str(image.parent.resolve()),
              "image_sha256": data["image_sha256"], "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
              "depth_sha256": bundle.observation.depth_sha256, "sequence": bundle.observation.sequence,
              "world_frame": bundle.observation.world_frame, "units": "meters",
              "mask_pixels": int(np.count_nonzero(mask)), "point_count": len(points),
              "valid_mask_depth_fraction": len(points) / int(np.count_nonzero(mask)),
              "depth_range_m": [float(depth.min()), float(depth.max())],
              "surface_centroid_m": points.mean(axis=0).tolist(),
              "world_aabb_min_m": minimum, "world_aabb_max_m": maximum,
              "obb": obb, "obb_extent_convention": "half-extents",
              "visible_surface_dimensions_m": dimensions, "obb_noise_seed": 0,
              "reference_deprojection_max_error_m": max(errors),
              "upstream_tools": ["geometry.mask_to_world_points", "geometry.compute_obb"],
              "skills_revision": SKILLS_REVISION, "grasp_geometry_validated": False,
              "limitations": ["Visible-surface centroid is not the full object center.",
                              "Single-view OBB can underestimate hidden dimensions.",
                              "No support completion, grasp planning, or robot execution performed."],
              "models_loaded": False, "network_connections_blocked": True, "motion_enabled": False}
    destination = output / "geometry.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"ok": True, "result": str(destination), "point_count": len(points),
            "surface_centroid_m": report["surface_centroid_m"],
            "visible_surface_dimensions_m": dimensions, "grasp_geometry_validated": False,
            "models_loaded": False, "motion_enabled": False}


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run(root, args.segmentation), indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__,
                          "detail": str(exc)[:1500], "models_loaded": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
