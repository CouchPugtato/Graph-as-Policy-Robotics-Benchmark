from pathlib import Path

import pytest

from gap_isaac.config import load_task
from gap_isaac.grocery_scene import (
    GROUND_POSITION_M,
    GROUND_SIZE_M,
    ROBOT_BASE_POSITION_M,
    WORK_SURFACE_Z_M,
    build_grocery_scene_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def test_grocery_scene_plan_is_deterministic_and_complete() -> None:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    first = build_grocery_scene_plan(task, seed=7, target_id="alphabet_soup")
    second = build_grocery_scene_plan(task, seed=7, target_id="alphabet_soup")

    assert first == second
    assert {item.object_id for item in first.objects} == {item.id for item in task.objects}
    assert first.target_id == "alphabet_soup"


def test_container_is_static_on_ground_and_objects_start_above_it() -> None:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    plan = build_grocery_scene_plan(task, seed=0)

    basket = next(item for item in plan.objects if item.is_container)
    assert basket.position_m[2] == WORK_SURFACE_Z_M == 0
    assert all(
        item.position_m[2] > WORK_SURFACE_Z_M for item in plan.objects if not item.is_container
    )


def test_robot_is_mounted_at_work_surface_height() -> None:
    assert ROBOT_BASE_POSITION_M == (0.0, 0.0, WORK_SURFACE_Z_M) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_dynamic_object_bottom_starts_three_mm_above_ground(seed):
    plan = build_grocery_scene_plan(
        load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml"), seed
    )
    floor_top = GROUND_POSITION_M[2] + GROUND_SIZE_M[2] / 2
    assert floor_top == WORK_SURFACE_Z_M == 0
    for item in plan.objects:
        if not item.is_container:
            assert item.position_m[2] - item.size_m[2] / 2 == pytest.approx(floor_top + 0.003)
