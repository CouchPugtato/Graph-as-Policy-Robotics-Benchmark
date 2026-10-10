"""Non-motion consistency checks for saved target and basket geometry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


def check_pair(target: dict, basket: dict) -> dict:
    from gap_isaac.grocery_scene import GEOMETRY
    if target["target_id"] != "alphabet_soup" or basket["target_id"] != "basket":
        raise ValueError("Expected alphabet_soup and basket geometry")
    # both objects must come from one capture in the same coordinate frame
    for key in ("bundle", "image_sha256", "depth_sha256", "sequence", "world_frame", "units", "skills_revision"):
        if target[key] != basket[key]:
            raise ValueError(f"Geometry capture/frame mismatch: {key}")
    if target["units"] != "meters" or target["world_frame"] != "world":
        raise ValueError("Expected world-frame metric geometry")
    diagnostics = {}
    for report in (target, basket):
        object_id = report["target_id"]
        if not report.get("ok") or report.get("motion_enabled"):
            raise ValueError("Expected successful non-motion geometry results")
        if report["point_count"] < 10 or report["valid_mask_depth_fraction"] < .95:
            raise ValueError("Insufficient valid masked depth")
        if report["reference_deprojection_max_error_m"] > 1e-5:
            raise ValueError("Deprojection check failed")
        dimensions = report["visible_surface_dimensions_m"]
        # known proxy dimensions reject implausible masks without supplying a pose
        nominal = GEOMETRY[object_id].size_m
        observed = sorted(dimensions[:2]) + [dimensions[2]]
        expected = sorted(nominal[:2]) + [nominal[2]]
        ratios = [actual / known for actual, known in zip(observed, expected, strict=True)]
        if not all(math.isfinite(value) and .7 <= value <= 1.3 for value in ratios):
            raise ValueError(f"{object_id} visible dimensions differ from proxy size by more than 30%")
        obb = report["obb"]
        if report["obb_extent_convention"] != "half-extents":
            raise ValueError("OBB half-extent contract required")
        if any(abs(2 * obb["extent"][axis] - dimensions[index]) > 1e-8
               for index, axis in enumerate(("x", "y", "z"))):
            raise ValueError("OBB extent/dimension mismatch")
        if not all(math.isfinite(obb["center"][axis]) for axis in ("x", "y", "z")):
            raise ValueError("Nonfinite OBB center")
        q = obb["orientation"]
        if not math.isclose(sum(q[axis] ** 2 for axis in ("w", "x", "y", "z")), 1, abs_tol=1e-4):
            raise ValueError("OBB quaternion must be normalized")
        bottom = obb["center"]["z"] - obb["extent"]["z"]
        if not -.02 <= bottom <= .04:
            raise ValueError(f"{object_id} fitted bottom is inconsistent with ground layout")
        diagnostics[object_id] = {"point_count": report["point_count"],
                                  "dimensions_m": dimensions, "nominal_proxy_dimensions_m": nominal,
                                  "dimension_ratios_sorted_xy": ratios, "fitted_bottom_z_m": bottom,
                                  "obb_center_m": obb["center"]}
    centers = [report["obb"]["center"] for report in (target, basket)]
    distance = math.hypot(centers[0]["x"] - centers[1]["x"], centers[0]["y"] - centers[1]["y"])
    if not .15 < distance < 1:
        raise ValueError("Target/basket separation is implausible for this bounded scene")
    return {"ok": True, "capture_consistent": True, "proxy_size_plausibility_passed": True,
            "comparison_source": "known synthetic proxy sizes, diagnostic only; not pose input",
            "objects": diagnostics, "target_basket_xy_distance_m": distance,
            "graph_motion_ready": False, "models_loaded": False, "motion_enabled": False,
            "remaining_checks": ["Validate basket opening and interior clearance, not just its outer OBB.",
                                 "Validate a grasp pose against visible geometry and robot kinematics.",
                                 "Check live stage/capture alignment before executing any cached geometry."],
            "identity_selection": "scene-config colors; autonomous VLM selection not validated"}


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--basket", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = check_pair(json.loads(args.target.read_text()), json.loads(args.basket.read_text()))
        result["inputs"] = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for path in (args.target, args.basket)]
        directory = root / "results/perception"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ("geometry-pair-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps({"ok": True, "result": str(path), "graph_motion_ready": False,
                          "target_basket_xy_distance_m": result["target_basket_xy_distance_m"]}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "detail": str(exc), "motion_enabled": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
