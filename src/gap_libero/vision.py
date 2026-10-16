"""Image driven grocery perception with explicit observable routing predicates.

The lightweight backend uses color and metric depth with optional local VLM selection.
It never reads simulator object poses or simulator segmentation identifiers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gap_isaac.cylinder_center import fit_center
from gap_isaac.graph_policy import NodeResult
from gap_isaac.observation_bundle import rgb_png


def color_mask(rgb, object_id):
    image = np.asarray(rgb, dtype=float) / 255
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image requires three channels")
    red, green, blue = image.transpose(2, 0, 1)
    if object_id == "alphabet_soup":
        return (blue > 1.4 * red) & (blue > 1.3 * green) & (blue > 0.15)
    if object_id == "basket":
        return (green > 1.3 * red) & (green > 1.25 * blue) & (green > 0.15)
    raise ValueError("no deployed visual description for this object")


def propose_candidates(rgb):
    image = np.asarray(rgb, dtype=float) / 255
    high, low = image.max(axis=2), image.min(axis=2)
    delta = np.maximum(high - low, 1e-9)
    dominant = image.argmax(axis=2)
    red, green, blue = image.transpose(2, 0, 1)
    hue = (
        np.select(
            (dominant == 0, dominant == 1),
            (((green - blue) / delta) % 6, (blue - red) / delta + 2),
            default=(red - green) / delta + 4,
        )
        / 6
    )
    mask = (high > 0.15) & ((high - low) > 0.25 * high)
    if int(mask.sum()) > 80000:
        raise ValueError("foreground exceeds the bounded candidate search budget")
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    components = []
    searched = 0
    height, width = mask.shape
    while remaining:
        searched += 1
        if searched > 128:
            raise ValueError("candidate search exceeded the component budget")
        start = min(remaining)
        seed_hue = hue[start]
        remaining.remove(start)
        pending, component = [start], []
        while pending:
            y, x = pending.pop()
            component.append((y, x))
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor not in remaining:
                    continue
                difference = abs(hue[neighbor] - seed_hue)
                if min(difference, 1 - difference) <= 0.04:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
        if len(component) < 12:
            continue
        points = np.asarray(component)
        y, x = points.T
        region = np.zeros_like(mask)
        region[y, x] = True
        components.append(
            {
                "box_xyxy": [
                    max(0, int(x.min()) - 3),
                    max(0, int(y.min()) - 3),
                    min(width, int(x.max()) + 4),
                    min(height, int(y.max()) + 4),
                ],
                "pixels": len(points),
                "mask": region,
            }
        )
        if len(components) > 6:
            raise ValueError("more than six visible candidates requires a different detector")
    components.sort(key=lambda item: (item["box_xyxy"][0], item["box_xyxy"][1]))
    for index, candidate in enumerate(components):
        candidate["id"] = chr(ord("A") + index)
    return components


def masked_world_points(bundle, mask):
    camera = bundle.to_gap_camera()
    depth = camera["depth"]
    if np.asarray(mask).shape != depth.shape:
        raise ValueError("mask and depth must share dimensions")
    valid = mask & np.isfinite(depth) & (depth >= 0.015) & (depth <= 20)
    y, x = np.nonzero(valid)
    z = depth[valid]
    intrinsic = camera["intrinsics"]
    points = np.column_stack(
        (
            (x - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (y - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        )
    )
    pose = bundle.observation.camera_pose
    q = pose.orientation
    axis = np.array([q.x, q.y, q.z])
    rotated = points + 2 * q.w * np.cross(axis, points) + 2 * np.cross(axis, np.cross(axis, points))
    return rotated + np.array([pose.position.x, pose.position.y, pose.position.z])


def fit_basket_walls(cloud, center, angle, basket_size=(0.24, 0.20, 0.15)):
    half = np.asarray(basket_size[:2]) / 2
    inner = half - 0.015
    wall = cloud[(cloud[:, 2] > 0.025) & (cloud[:, 2] < 0.13), :2]
    if len(wall) < 40:
        raise ValueError("insufficient visible basket wall evidence")
    initial = np.r_[center, angle]
    rim = cloud[cloud[:, 2] >= np.quantile(cloud[:, 2], 0.98) - 0.0005, :2]

    def residual(parameters):
        x, y, yaw = parameters
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        local = (wall - [x, y]) @ rotation
        # fit inner and outer surfaces using the existing basket thickness prior
        distances = np.column_stack(
            [
                local[:, axis] - sign * size
                for axis, sizes in enumerate(zip(half, inner, strict=True))
                for size in sizes
                for sign in (-1, 1)
            ]
        )
        choice = np.abs(distances).argmin(axis=1)
        errors = distances[np.arange(len(wall)), choice]
        top = (rim - [x, y]) @ rotation
        outside = np.maximum(np.abs(top) - half, 0).max(axis=1)
        inside = np.maximum(
            np.minimum(inner[0] - np.abs(top[:, 0]), inner[1] - np.abs(top[:, 1])), 0
        )
        rim_midpoint = (top.min(axis=0) + top.max(axis=0)) / 2
        return np.r_[errors, outside, inside, rim_midpoint * np.sqrt(len(wall)) * 0.01], choice

    solutions = []
    for dx in (-0.01, 0, 0.01):
        for dy in (-0.01, 0, 0.01):
            parameters = initial + [dx, dy, 0]
            for _ in range(30):
                errors, _ = residual(parameters)
                jacobian = np.column_stack(
                    [
                        (residual(parameters + np.eye(3)[axis] * 1e-6)[0] - errors) / 1e-6
                        for axis in range(3)
                    ]
                )
                weights = np.minimum(1, 0.0015 / np.maximum(np.abs(errors), 1e-12))
                change = np.linalg.lstsq(
                    jacobian * np.sqrt(weights[:, None]), errors * np.sqrt(weights), rcond=None
                )[0]
                change *= min(
                    1,
                    0.003 / max(np.linalg.norm(change[:2]), 1e-12),
                    0.01 / max(abs(change[2]), 1e-12),
                )
                parameters -= change
                if (
                    np.linalg.norm(parameters[:2] - initial[:2]) > 0.02
                    or abs(parameters[2] - initial[2]) > 0.08
                ):
                    break
                if np.linalg.norm(change) < 1e-9:
                    break
            if (
                np.linalg.norm(parameters[:2] - initial[:2]) <= 0.02
                and abs(parameters[2] - initial[2]) <= 0.08
            ):
                errors, _ = residual(parameters)
                cost = float(np.mean(np.minimum(errors**2, 0.003**2)))
                cost += 0.0001 * float(np.sum((parameters[:2] - initial[:2]) ** 2))
                solutions.append((cost, parameters))
    if not solutions:
        raise ValueError("basket wall and rim estimates disagree")
    _, parameters = min(solutions, key=lambda item: item[0])
    errors, choice = residual(parameters)
    errors = errors[: len(wall)]
    inliers = np.abs(errors) <= 0.0015
    counts = [int(np.count_nonzero(inliers & (choice // 4 == axis))) for axis in (0, 1)]
    rotation = np.array(
        [
            [np.cos(parameters[2]), -np.sin(parameters[2])],
            [np.sin(parameters[2]), np.cos(parameters[2])],
        ]
    )
    local_rim = (rim - parameters[:2]) @ rotation
    rim_counts = [
        [
            int(
                np.count_nonzero(
                    (np.abs(local_rim[:, axis] - sign * edge) < 0.008)
                    & (np.abs(local_rim[:, 1 - axis]) < half[1 - axis] - 0.02)
                )
            )
            for sign in (-1, 1)
        ]
        for axis, edge in enumerate(half)
    ]
    rms = float(np.sqrt(np.mean(errors[inliers] ** 2))) if inliers.any() else float("inf")
    constrained = [counts[axis] >= 20 or min(rim_counts[axis]) >= 20 for axis in (0, 1)]
    if not all(constrained) or inliers.mean() < 0.65 or rms > 0.001:
        raise ValueError("basket wall fit lacks independent plane support")
    return (
        parameters[:2],
        float(parameters[2]),
        {
            "revision": "joint-inner-outer-walls-v1",
            "axis_support": counts,
            "opposite_rim_support": rim_counts,
            "inlier_fraction": float(inliers.mean()),
            "rms_error_m": rms,
            "wall_thickness_prior_m": 0.015,
            "center_shift_m": float(np.linalg.norm(parameters[:2] - initial[:2])),
        },
    )


def estimate_geometry(
    points,
    object_id,
    camera_position=None,
    *,
    silhouette_points=None,
    basket_size=(0.24, 0.20, 0.15),
):
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 30:
        raise ValueError("at least thirty world points required")
    cloud_limit = (
        max(0.4, float(np.linalg.norm(basket_size[:2])) + 0.01) if object_id == "basket" else 0.4
    )
    if not np.isfinite(cloud).all() or np.any(np.ptp(cloud, axis=0) > cloud_limit):
        raise ValueError("invalid or unbounded object cloud")
    if cloud[:, 2].min() < -0.02:
        raise ValueError("object points below the support plane")
    if object_id == "alphabet_soup":
        top = np.quantile(cloud[:, 2], 0.98)
        side = cloud[(cloud[:, 2] > top - 0.095) & (cloud[:, 2] < top - 0.01), :2]
        if len(side) < 30:
            raise ValueError("insufficient cylinder sidewall evidence")
        design = np.column_stack((2 * side, np.ones(len(side))))
        if np.linalg.matrix_rank(design) != 3:
            raise ValueError("cylinder sidewall is geometrically degenerate")
        initial = np.linalg.lstsq(design, np.sum(side * side, axis=1), rcond=None)[0][:2]
        if np.linalg.norm(initial - side.mean(axis=0)) > 0.06:
            raise ValueError("cylinder initialization is outside the visible object")
        top_rim = None
        if silhouette_points is not None:
            silhouette = np.asarray(silhouette_points, dtype=float)
            if silhouette.ndim != 2 or silhouette.shape[1] != 3:
                raise ValueError("invalid silhouette cloud")
            cap = silhouette[
                np.isfinite(silhouette).all(axis=1) & (np.abs(silhouette[:, 2] - top) < 0.0005)
            ]
            if len(cap) >= 8:
                directions = np.linspace(-np.pi, np.pi, 128, endpoint=False)
                support = cap[:, :2] @ np.column_stack((np.cos(directions), np.sin(directions))).T
                top_rim = cap[np.unique(support.argmax(axis=0))]
        fit = fit_center(cloud, initial, top_rim=top_rim)
        height = float(np.quantile(cloud[:, 2], 0.98))
        if not 0.085 <= height <= 0.14:
            raise ValueError("target height inconsistent with upright cylinder prior")
        return {
            "position_m": [*fit["center_xy_m"], height - 0.055],
            "yaw_rad": 0.0,
            "point_count": len(cloud),
            "fit": fit,
            "uses_evaluator_pose": False,
            "geometry_source": "masked-rgbd",
        }
    if object_id != "basket":
        raise ValueError("unknown geometry prior")
    rim = cloud[cloud[:, 2] >= np.quantile(cloud[:, 2], 0.65)]
    best = None
    for angle in np.linspace(-np.pi / 2, np.pi / 2, 361, endpoint=False):
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        local = rim[:, :2] @ rotation
        low, high = local.min(axis=0), local.max(axis=0)
        spans = high - low
        area = float(np.prod(spans))
        if best is None or area < best[0]:
            best = (area, spans, (low + high) / 2 @ rotation.T, angle)
    _, spans, center, angle = best
    expected = sorted(basket_size[:2])
    long_limits, short_limits = (
        ((0.17, 0.27), (0.14, 0.23))
        if tuple(basket_size) == (0.24, 0.20, 0.15)
        else ((0.7 * expected[1], 1.125 * expected[1]), (0.7 * expected[0], 1.15 * expected[0]))
    )
    if not (
        long_limits[0] <= max(spans) <= long_limits[1]
        and short_limits[0] <= min(spans) <= short_limits[1]
    ):
        raise ValueError("basket rim is incomplete or inconsistent with its size prior")
    if spans[0] < spans[1]:
        angle += np.pi / 2
    if camera_position is not None:
        center, angle, wall_fit = fit_basket_walls(cloud, center, angle, basket_size)
    return {
        "position_m": [*center.tolist(), 0.0],
        "yaw_rad": float(angle),
        "point_count": len(cloud),
        "rim_dimensions_m": sorted(spans.tolist(), reverse=True),
        "wall_fit": wall_fit if camera_position is not None else None,
        "uses_evaluator_pose": False,
        "geometry_source": "masked-rgbd",
    }


class VisionActions:
    def __init__(
        self,
        capture,
        output: Path,
        *,
        identifier=None,
        descriptions=None,
        geometry_estimator=None,
        silhouette_objects=None,
    ):
        self.capture = capture
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        self.bundle = None
        self.masks = {}
        self.geometries = {}
        self.counter = 0
        self.identifier = identifier
        self.candidates = None
        self.identifications = {}
        self.descriptions = descriptions or {}
        self.geometry_estimator = geometry_estimator or estimate_geometry
        self.silhouette_objects = (
            {"alphabet_soup"} if silhouette_objects is None else set(silhouette_objects)
        )

    def observe(self, _state):
        self.bundle = self.capture()
        self.masks.clear()
        self.geometries.clear()
        self.identifications.clear()
        self.candidates = None
        directory = self.output / f"observation-{self.counter:03d}"
        self.counter += 1
        self.bundle.save(directory)
        return NodeResult(
            outputs={
                "observation_path": str(directory),
                "rgb_sha256": self.bundle.observation.rgb_sha256,
                "depth_sha256": self.bundle.observation.depth_sha256,
            }
        )

    def detect(self, object_id, _state):
        if self.bundle is None:
            raise RuntimeError("observe before detecting")
        camera = self.bundle.to_gap_camera()
        if self.identifier is not None:
            if self.candidates is None:
                self.candidates = propose_candidates(camera["rgb"])
            return NodeResult(
                outputs={
                    object_id + "_detection": {
                        "backend": "color-region-proposals",
                        "rgb_sha256": self.bundle.observation.rgb_sha256,
                        "candidates": [
                            {key: value for key, value in item.items() if key != "mask"}
                            for item in self.candidates
                        ],
                    }
                }
            )
        mask = color_mask(camera["rgb"], object_id)
        self.masks[object_id] = mask
        y, x = np.nonzero(mask)
        box = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if len(x) else None
        return NodeResult(
            outputs={
                object_id + "_detection": {
                    "box_xyxy": box,
                    "pixels": len(x),
                    "backend": "rgbd-color",
                    "rgb_sha256": self.bundle.observation.rgb_sha256,
                }
            }
        )

    def identify(self, object_id, _state):
        if self.identifier is None or self.candidates is None:
            raise RuntimeError("local VLM identification requires detection first")
        descriptions = {
            "alphabet_soup": "the designated blue upright cylinder proxy object",
            "basket": "the green open basket or open box container",
        }
        descriptions.update(self.descriptions)
        if self.candidates:
            path = self.output / f"{self.counter:03d}-{object_id}-identification.json"
            report = self.identifier.identify(
                self.bundle.to_gap_camera()["rgb"], self.candidates, descriptions[object_id], path
            )
        else:
            report = {
                "candidate_id": None,
                "reason": "no visible object candidates",
                "model": self.identifier.model,
                "local_only": True,
                "inference_performed": False,
            }
        selected = next(
            (item for item in self.candidates if item["id"] == report["candidate_id"]), None
        )
        self.identifications[object_id] = report
        self.masks[object_id] = (
            selected["mask"].copy()
            if selected is not None
            else np.zeros(self.bundle.to_gap_camera()["rgb"].shape[:2], dtype=bool)
        )
        return NodeResult(
            outputs={
                object_id + "_identification": report,
                object_id + "_identified": selected is not None,
            }
        )

    def segment(self, object_id, _state):
        if self.identifier is not None and object_id not in self.identifications:
            raise RuntimeError("local VLM must identify the object before segmentation")
        mask = self.masks[object_id]
        rgb = np.repeat((mask * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        destination = self.output / f"{self.counter:03d}-{object_id}-mask.png"
        destination.write_bytes(rgb_png(rgb.tobytes(), rgb.shape[1], rgb.shape[0]))
        return NodeResult(
            outputs={
                object_id + "_mask": str(destination),
                object_id + "_mask_pixels": int(mask.sum()),
            }
        )

    def geometry(self, object_id, _state):
        mask = self.masks[object_id]
        core = np.zeros_like(mask)
        core[1:-1, 1:-1] = (
            mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        )
        points = masked_world_points(self.bundle, core)
        cloud_path = self.output / f"{self.counter:03d}-{object_id}-points.npy"
        np.save(cloud_path, points, allow_pickle=False)
        try:
            pose = self.bundle.observation.camera_pose.position
            silhouette = (
                masked_world_points(self.bundle, mask)
                if object_id in self.silhouette_objects
                else None
            )
            geometry = self.geometry_estimator(
                points, object_id, [pose.x, pose.y, pose.z], silhouette_points=silhouette
            )
        except ValueError as error:
            geometry = {"valid": False, "detail": str(error), "point_count": len(points)}
        else:
            geometry["valid"] = True
        geometry.update(
            rgb_sha256=self.bundle.observation.rgb_sha256,
            depth_sha256=self.bundle.observation.depth_sha256,
            cloud_path=str(cloud_path),
        )
        self.geometries[object_id] = geometry
        path = self.output / f"{self.counter:03d}-{object_id}-geometry.json"
        path.write_text(json.dumps(geometry, indent=2), encoding="utf-8")
        return NodeResult(outputs={object_id + "_geometry": geometry})

    def found(self, object_id, _state):
        valid = self.geometries.get(object_id, {}).get("valid") is True
        if self.identifier is not None:
            valid = valid and (
                self.identifications.get(object_id, {}).get("candidate_id") is not None
            )
        detail = (
            "image geometry verified"
            if valid
            else self.geometries.get(object_id, {}).get("detail", "no image geometry available")
        )
        return NodeResult(
            "true" if valid else "false", detail, outputs={object_id + "_found": valid}
        )

    def registry(self):
        actions = {"observe_scene": self.observe}
        for prefix, object_id in (("target", "alphabet_soup"), ("container", "basket")):
            for verb, function in (
                ("detect", self.detect),
                ("segment", self.segment),
                ("estimate", self.geometry),
                ("check", self.found),
            ):
                actions[f"{verb}_{prefix}"] = lambda state, fn=function, key=object_id: fn(
                    key, state
                )
            if self.identifier is not None:
                actions[f"identify_{prefix}"] = lambda state, key=object_id: self.identify(
                    key, state
                )
        return actions
