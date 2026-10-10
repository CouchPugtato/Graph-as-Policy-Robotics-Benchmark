from pathlib import Path

import pytest

from gap_isaac.config import load_task
from gap_isaac.evaluate import ObjectState, target_fits_basket_opening, target_is_contained
from gap_isaac.grocery_scene import GEOMETRY, WORK_SURFACE_Z_M
from gap_isaac.pick_place_plan import build_oracle_pick_place_plan
from gap_isaac.reset import build_episode_reset_plan

ROOT = Path(__file__).resolve().parents[1]


def test_oracle_plan_aligns_with_target_and_releases_above_basket() -> None:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    reset = build_episode_reset_plan(task, seed=0, target_id="alphabet_soup")
    plan = build_oracle_pick_place_plan(task, reset)
    objects = {item.object_id: item for item in reset.objects}
    target = objects[reset.target_id]
    basket = objects[reset.container_id]

    assert plan.target_center.as_tuple() == target.position_m
    assert plan.grasp.z_m == pytest.approx(target.position_m[2] + 0.025)
    assert plan.align.x_m == target.position_m[0]
    assert plan.align.y_m == target.position_m[1]
    assert plan.transport.x_m == basket.position_m[0]
    assert plan.transport.y_m == basket.position_m[1]
    assert plan.align.z_m >= WORK_SURFACE_Z_M + 0.30
    assert plan.lift.z_m == plan.transport.z_m == plan.retract.z_m

    placed_object_center = (
        plan.place.x_m,
        plan.place.y_m,
        plan.place.z_m - plan.grasp_frame_above_object_center_m,
    )
    placed = ObjectState(reset.target_id, placed_object_center)
    container = ObjectState(reset.container_id, basket.position_m, basket.yaw_deg)
    assert target_fits_basket_opening(placed, container, task.evaluation)
    assert not target_is_contained(placed, container, task.evaluation)
    bottom_z = placed.position_m[2] - GEOMETRY[reset.target_id].size_m[2] / 2
    assert bottom_z - plan.basket_rim_z_m == pytest.approx(0.04)
    assert plan.transport.z_m >= plan.place.z_m


def test_oracle_place_height_uses_target_geometry() -> None:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    reset = build_episode_reset_plan(task, seed=1, target_id="cream_cheese")
    plan = build_oracle_pick_place_plan(task, reset)
    basket = next(item for item in reset.objects if item.object_id == reset.container_id)

    expected = (
        basket.position_m[2]
        + task.evaluation.basket_base_thickness_m
        + GEOMETRY[reset.container_id].size_m[2]
        + GEOMETRY[reset.target_id].size_m[2] / 2
        + 0.04
    )
    assert plan.place.z_m - plan.grasp_frame_above_object_center_m == pytest.approx(expected)


def test_oracle_plan_uses_settled_positions_when_supplied() -> None:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    reset = build_episode_reset_plan(task, seed=0, target_id="alphabet_soup")
    settled = {
        "alphabet_soup": (0.61, 0.12, 0.055),
        "basket": (0.42, -0.21, 0.0),
    }

    plan = build_oracle_pick_place_plan(task, reset, settled)

    assert plan.target_center.as_tuple() == settled["alphabet_soup"]
    assert plan.transport.x_m == settled["basket"][0]
    assert plan.transport.y_m == settled["basket"][1]


@pytest.mark.parametrize(
    "target_id", ["alphabet_soup", "chocolate_pudding", "tomato_sauce", "cream_cheese"]
)
def test_all_proxy_release_waypoints_clear_the_actual_wall_top(target_id) -> None:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    reset = build_episode_reset_plan(task, seed=0, target_id=target_id)
    plan = build_oracle_pick_place_plan(task, reset)
    basket = next(obj for obj in reset.objects if obj.object_id == reset.container_id)
    assert plan.basket_rim_z_m == pytest.approx(basket.position_m[2] + 0.02 + 0.15)
    target_bottom = (
        plan.place.z_m - plan.grasp_frame_above_object_center_m - GEOMETRY[target_id].size_m[2] / 2
    )
    assert target_bottom >= plan.basket_rim_z_m + 0.04 - 1e-12
