"""Shape prior fits using only local Qwen selected image masks and camera depth."""

import numpy as np

from gap_libero.scenario_specs import PRIORS
from gap_libero.vision import estimate_geometry


def fit_cylinder_cap(cloud, silhouette, size):
    top = float(np.quantile(cloud[:, 2], 0.98))
    source = cloud if silhouette is None else np.asarray(silhouette, dtype=float)
    cap = source[np.isfinite(source).all(axis=1) & (np.abs(source[:, 2] - top) < 0.0008), :2]
    radius = size[0] / 2
    if len(cap) < 20 or not 0.085 <= top <= 0.14:
        raise ValueError("insufficient upright cylinder cap evidence")
    spans = np.ptp(cap, axis=0)
    if np.any(spans < 1.4 * radius) or np.any(spans > 2.2 * radius):
        raise ValueError("cylinder cap coverage does not constrain its center")
    directions = np.linspace(-np.pi, np.pi, 128, endpoint=False)
    support = cap @ np.column_stack((np.cos(directions), np.sin(directions))).T
    rim = cap[np.unique(support.argmax(axis=0))]
    if len(rim) < 8:
        raise ValueError("cylinder cap lacks independent rim evidence")
    center = (cap.min(axis=0) + cap.max(axis=0)) / 2
    for _ in range(30):
        delta = center - rim
        distance = np.linalg.norm(delta, axis=1)
        step = np.linalg.lstsq(
            delta / np.maximum(distance[:, None], 1e-9), distance - radius, rcond=None
        )[0]
        center -= step
        if np.linalg.norm(step) < 1e-8:
            break
    error = float(np.sqrt(np.mean((np.linalg.norm(rim - center, axis=1) - radius) ** 2)))
    if error > 0.002 or np.linalg.norm(center - cap.mean(axis=0)) > 0.015:
        raise ValueError("cylinder cap does not support the declared radius")
    return {
        "position_m": [*center.tolist(), top - size[2] / 2],
        "yaw_rad": 0.0,
        "point_count": len(cloud),
        "shape_prior_m": list(size),
        "cap_fit": {
            "method": "visible-top-rim-known-radius",
            "rms_error_m": error,
            "rim_points": len(rim),
        },
        "uses_evaluator_pose": False,
        "geometry_source": "qwen-selected-masked-rgbd",
    }


def estimate(
    points,
    object_id,
    camera_position=None,
    *,
    silhouette_points=None,
    basket_size=(0.24, 0.20, 0.15),
):
    if object_id == "basket":
        return estimate_geometry(
            points,
            object_id,
            camera_position,
            silhouette_points=silhouette_points,
            basket_size=basket_size,
        )
    shape, size = PRIORS[object_id]
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 30 or not np.isfinite(cloud).all():
        raise ValueError("insufficient finite object depth evidence")
    if np.any(np.ptp(cloud, axis=0) > 0.4) or cloud[:, 2].min() < -0.02:
        raise ValueError("object cloud violates the bounded workcell")
    if shape == "cylinder":
        scale = 0.07 / size[0]
        try:
            fit = estimate_geometry(
                cloud * scale,
                "alphabet_soup",
                silhouette_points=None
                if silhouette_points is None
                else np.asarray(silhouette_points) * scale,
            )
        except ValueError:
            return fit_cylinder_cap(cloud, silhouette_points, size)
        fit["position_m"] = [
            fit["position_m"][0] / scale,
            fit["position_m"][1] / scale,
            float(np.quantile(cloud[:, 2], 0.98)) - size[2] / 2,
        ]
        fit["shape_prior_m"] = list(size)
        return fit
    top = float(np.quantile(cloud[:, 2], 0.98))
    surface = cloud if silhouette_points is None else np.asarray(silhouette_points)
    cap = surface[np.isfinite(surface).all(axis=1) & (np.abs(surface[:, 2] - top) < 0.001)]
    if len(cap) < 20:
        raise ValueError("insufficient flat top surface evidence")
    candidates = []
    for angle in np.linspace(-np.pi / 2, np.pi / 2, 361, endpoint=False):
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        local = cap[:, :2] @ rotation
        low, high = local.min(axis=0), local.max(axis=0)
        spans = high - low
        candidates.append((float(np.prod(spans)), spans, (low + high) / 2 @ rotation.T, angle))
    _, spans, center, yaw = min(candidates, key=lambda item: item[0])
    if not np.all(
        (np.sort(spans) / np.sort(size[:2]) >= 0.65) & (np.sort(spans) / np.sort(size[:2]) <= 1.1)
    ):
        raise ValueError("top surface does not support the declared box size")
    if spans[0] < spans[1]:
        yaw += np.pi / 2
    return {
        "position_m": [*center.tolist(), top - size[2] / 2],
        "yaw_rad": float(yaw),
        "point_count": len(cloud),
        "shape_prior_m": list(size),
        "visible_top_dimensions_m": spans.tolist(),
        "uses_evaluator_pose": False,
        "geometry_source": "qwen-selected-masked-rgbd",
    }
