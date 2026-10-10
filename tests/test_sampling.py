import math
from pathlib import Path

import pytest

from gap_isaac.config import load_task
from gap_isaac.sampling import sample_instance

ROOT = Path(__file__).resolve().parents[1]


def test_sampling_is_deterministic_and_non_overlapping() -> None:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    first = sample_instance(task, 17)
    second = sample_instance(task, 17)
    assert first == second

    specs = {item.id: item for item in task.objects}
    for index, left in enumerate(first.poses):
        for right in first.poses[index + 1 :]:
            distance = math.hypot(left.x_m - right.x_m, left.y_m - right.y_m)
            required = (
                specs[left.object_id].footprint_radius_m
                + specs[right.object_id].footprint_radius_m
                + task.minimum_clearance_m
            )
            assert distance >= required


def test_sampling_rejects_unknown_target() -> None:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    with pytest.raises(ValueError, match="not declared"):
        sample_instance(task, 0, target_id="unknown")
