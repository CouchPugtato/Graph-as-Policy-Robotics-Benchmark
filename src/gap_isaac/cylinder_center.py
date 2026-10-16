"""CPU-only, bounded XY fit for the known upright alphabet_soup cylinder.

Raw OBB and capture checks stay authoritative. This fits saved surface points;
it accepts no simulator state, seed, evaluator pose, or directional offset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REVISION = "upright-cylinder-center-v1"
RADIUS_M = .035
MAX_SHIFT_M = .015


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_center(points, initial_xy, *, top_rim=None) -> dict:
    import numpy as np

    cloud = np.asarray(points, dtype=float)
    initial = np.asarray(initial_xy, dtype=float)
    if (cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 30
            or initial.shape != (2,) or not np.isfinite(cloud).all()
            or not np.isfinite(initial).all()):
        raise ValueError("Cylinder fit requires finite XYZ points and an XY center")
    # Exclude top and support surfaces. The 110 mm height is the existing proxy prior.
    near = cloud[np.linalg.norm(cloud[:, :2] - initial, axis=1) < .060]
    if len(near) < 30:
        raise ValueError("Insufficient local cylinder surface")
    top = float(np.quantile(near[:, 2], .95))
    side = near[(near[:, 2] > top - .110 + .015) & (near[:, 2] < top - .010), :2]
    if len(side) < 24:
        raise ValueError("Insufficient cylinder sidewall points")
    rng = np.random.default_rng(0)
    # fit the known cylinder radius using only the saved visible surface
    candidates = [initial]
    for a, b in rng.integers(0, len(side), size=(1024, 2)):
        delta = side[b] - side[a]
        distance = float(np.linalg.norm(delta))
        if not .015 < distance < 2 * RADIUS_M:
            continue
        midpoint = (side[a] + side[b]) / 2
        normal = np.array([-delta[1], delta[0]]) / distance
        height = (RADIUS_M ** 2 - (distance / 2) ** 2) ** .5
        for candidate in (midpoint + normal * height, midpoint - normal * height):
            if np.linalg.norm(candidate - initial) <= MAX_SHIFT_M:
                candidates.append(candidate)
    def score(center):
        errors = np.abs(np.linalg.norm(side - center, axis=1) - RADIUS_M)
        return (int(np.count_nonzero(errors <= .0015)), -float(np.median(errors)))
    # refine the candidate supported by the most sidewall points
    center = max(candidates, key=score).copy()
    for _ in range(12):
        vectors = center - side
        radii = np.linalg.norm(vectors, axis=1)
        residual = radii - RADIUS_M
        mask = (np.abs(residual) <= .0015) & (radii > 1e-9)
        if np.count_nonzero(mask) < 20:
            raise ValueError("Insufficient cylinder fit inliers")
        jacobian = vectors[mask] / radii[mask, None]
        delta = np.linalg.lstsq(jacobian, residual[mask], rcond=None)[0]
        center -= delta
        if np.linalg.norm(delta) < 1e-9:
            break
    residual = np.linalg.norm(side - center, axis=1) - RADIUS_M
    mask = np.abs(residual) <= .0015
    inliers = side[mask]
    if len(inliers) < 20 or len(inliers) / len(side) < .65:
        raise ValueError("Cylinder fit lacks sidewall consensus")
    angles = np.sort(np.arctan2(inliers[:, 1] - center[1], inliers[:, 0] - center[0]))
    span = float(2 * np.pi - np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])))
    side_span = span
    rim_support = None
    if span < np.pi / 3 and top_rim is not None:
        rim = np.asarray(top_rim, dtype=float)
        if (rim.ndim != 2 or rim.shape[1] != 3 or not 8 <= len(rim) <= 128
                or not np.isfinite(rim).all() or np.ptp(rim[:, 2]) > .001
                or abs(np.median(rim[:, 2]) - top) > .002):
            raise ValueError("Cylinder top rim evidence is invalid")
        radial = np.linalg.norm(rim[:, :2] - center, axis=1) - RADIUS_M
        supported = np.abs(radial) <= .0015
        if np.count_nonzero(supported) >= 8 and supported.mean() >= .65:
            rim_angles = np.sort(np.arctan2(rim[supported, 1] - center[1],
                                           rim[supported, 0] - center[0]))
            rim_span = float(2 * np.pi - np.max(np.diff(
                np.r_[rim_angles, rim_angles[0] + 2 * np.pi])))
            if rim_span >= np.pi / 3:
                span = rim_span
                rim_support = {"points": int(supported.sum()), "angular_span_rad": rim_span,
                    "rms_radial_error_m": float(np.sqrt(np.mean(radial[supported] ** 2)))}
    rms = float(np.sqrt(np.mean(residual[mask] ** 2)))
    shift = float(np.linalg.norm(center - initial))
    split_centers = []
    for subset in (inliers[::2], inliers[1::2]):
        trial = center.copy()
        for _ in range(8):
            vectors = trial - subset
            radii = np.linalg.norm(vectors, axis=1)
            trial -= np.linalg.lstsq(vectors / radii[:, None], radii - RADIUS_M,
                                     rcond=None)[0]
        split_centers.append(trial)
    stability = float(np.linalg.norm(split_centers[0] - split_centers[1]))
    # reject weak fits instead of silently moving the grasp center
    if span < np.pi / 3 or rms > .001 or shift > MAX_SHIFT_M or stability > .0015:
        raise ValueError("Cylinder fit is inconsistent or insufficiently constrained")
    return {"revision": REVISION, "center_xy_m": center.tolist(), "radius_prior_m": RADIUS_M,
            "height_prior_m": .110, "initial_xy_m": initial.tolist(), "shift_m": shift,
            "sidewall_points": len(side), "inlier_count": len(inliers),
            "inlier_fraction": len(inliers) / len(side), "rms_radial_error_m": rms,
            "angular_span_rad": span, "split_center_distance_m": stability,
            "sidewall_angular_span_rad": side_span, "top_rim_support": rim_support,
            "uses_evaluator_pose": False}


def augment(target_path: Path, basket_path: Path, output: Path) -> dict:
    import numpy as np

    from gap_isaac.geometry_pair_check import check_pair

    target = json.loads(target_path.read_text(encoding="utf-8"))
    basket = json.loads(basket_path.read_text(encoding="utf-8"))
    check_pair(target, basket)  # Unchanged gate runs on the ORIGINAL measurements.
    points_path = target_path.parent / "world_points.npy"
    fit = fit_center(np.load(points_path, allow_pickle=False),
                     [target["obb"]["center"][axis] for axis in ("x", "y")])
    fit.update(source_geometry=str(target_path.resolve()), source_sha256=digest(target_path),
               world_points=str(points_path.resolve()), world_points_sha256=digest(points_path))
    target["cylinder_center"] = fit
    with output.open("x", encoding="utf-8") as stream:
        json.dump(target, stream, indent=2)
    return fit


def verified_center(target: dict) -> tuple[float, float]:
    import math

    fit = target.get("cylinder_center", {})
    if fit.get("revision") != REVISION or fit.get("uses_evaluator_pose") is not False:
        raise ValueError("A cylinder-center-v1 receipt without evaluator pose input is required")
    source = Path(fit["source_geometry"])
    points = Path(fit["world_points"])
    if digest(source) != fit["source_sha256"] or digest(points) != fit["world_points_sha256"]:
        raise ValueError("Cylinder fit source hashes changed")
    raw = {key: value for key, value in target.items() if key != "cylinder_center"}
    if raw != json.loads(source.read_text(encoding="utf-8")):
        raise ValueError("Cylinder receipt changed original geometry")
    # Recompute the small CPU fit so edited receipt metrics cannot authorize a pose.
    import numpy as np
    expected = fit_center(np.load(points, allow_pickle=False),
                          [raw["obb"]["center"][axis] for axis in ("x", "y")])
    def equal(a, b):
        if isinstance(a, float) and isinstance(b, (float, int)):
            return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b
    if any(not equal(expected[key], fit.get(key)) for key in expected):
        raise ValueError("Cylinder receipt differs from deterministic surface fit")
    xy = tuple(float(v) for v in fit["center_xy_m"])
    if len(xy) != 2 or not all(map(math.isfinite, xy)):
        raise ValueError("Invalid fitted center")
    return xy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--basket", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        fit = augment(args.target, args.basket, args.output)
        print(json.dumps({"ok": True, "result": str(args.output.resolve()), "fit": fit}))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
