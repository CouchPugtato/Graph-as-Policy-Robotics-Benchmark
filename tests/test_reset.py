from pathlib import Path

from gap_isaac.config import load_task
from gap_isaac.reset import FRANKA_JOINT_NAMES, build_episode_reset_plan

ROOT = Path(__file__).resolve().parents[1]


def test_reset_plan_is_deterministic_and_complete() -> None:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    first = build_episode_reset_plan(task, seed=13, target_id="tomato_sauce")
    second = build_episode_reset_plan(task, seed=13, target_id="tomato_sauce")

    assert first == second
    assert first.target_id == "tomato_sauce"
    assert first.container_id == "basket"
    assert first.robot_joint_names == FRANKA_JOINT_NAMES
    assert len(first.robot_joint_positions_rad) == len(FRANKA_JOINT_NAMES)
    assert {item.object_id for item in first.objects} == {item.id for item in task.objects}
    assert all(item.linear_velocity_m_s == (0.0, 0.0, 0.0) for item in first.objects)
    assert all(item.angular_velocity_rad_s == (0.0, 0.0, 0.0) for item in first.objects)


def test_reset_plan_changes_with_seed() -> None:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    first = build_episode_reset_plan(task, seed=1)
    second = build_episode_reset_plan(task, seed=2)

    assert first.objects != second.objects
