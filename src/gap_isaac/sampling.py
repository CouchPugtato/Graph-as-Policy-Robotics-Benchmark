"""Deterministic collision-aware XY sampling for grocery task instances."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from gap_isaac.config import ObjectSpec, TaskConfig


@dataclass(frozen=True)
class SampledPose:
    object_id: str
    x_m: float
    y_m: float
    yaw_deg: float


@dataclass(frozen=True)
class TaskInstance:
    seed: int
    target_id: str
    poses: tuple[SampledPose, ...]

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "target_id": self.target_id,
            "poses": [asdict(pose) for pose in self.poses],
        }


def _overlaps(
    candidate: SampledPose,
    candidate_spec: ObjectSpec,
    poses: list[SampledPose],
    specs: dict[str, ObjectSpec],
    clearance_m: float,
) -> bool:
    for existing in poses:
        required = (
            candidate_spec.footprint_radius_m
            + specs[existing.object_id].footprint_radius_m
            + clearance_m
        )
        if math.hypot(candidate.x_m - existing.x_m, candidate.y_m - existing.y_m) < required:
            return True
    return False


def sample_instance(task: TaskConfig, seed: int, target_id: str | None = None) -> TaskInstance:
    """Sample one reproducible, non-overlapping scene instance."""

    # the same seed reproduces the same object layout
    rng = random.Random(seed)
    selected_target = target_id or task.target_ids[seed % len(task.target_ids)]
    if selected_target not in task.target_ids:
        raise ValueError(f"target is not declared by task: {selected_target}")

    specs = {item.id: item for item in task.objects}
    poses: list[SampledPose] = []
    for item in task.objects:
        # reject overlapping placements without retrying forever
        for _ in range(task.max_sampling_attempts):
            candidate = SampledPose(
                object_id=item.id,
                x_m=rng.uniform(*task.spawn.xy_m.x),
                y_m=rng.uniform(*task.spawn.xy_m.y),
                yaw_deg=rng.uniform(*task.spawn.yaw_deg),
            )
            if not _overlaps(candidate, item, poses, specs, task.minimum_clearance_m):
                poses.append(candidate)
                break
        else:
            raise RuntimeError(
                f"could not place {item.id} after {task.max_sampling_attempts} attempts"
            )

    return TaskInstance(seed=seed, target_id=selected_target, poses=tuple(poses))
