from pathlib import Path
from unittest.mock import Mock

import pytest

from gap_isaac.config import load_task
from gap_isaac.evaluate import EpisodeStatus, PrivilegedEpisodeEvaluator
from gap_isaac.isaac_episode import IsaacEpisodeRuntime
from gap_isaac.reset import build_episode_reset_plan

ROOT = Path(__file__).resolve().parents[1]


def setup_runtime() -> tuple:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    reset = build_episode_reset_plan(task, seed=0, target_id="alphabet_soup")
    robot = Mock()
    robot.dof_names = list(reversed(reset.robot_joint_names))
    prims = {}
    for state in reset.objects:
        prim = Mock()
        prim.get_world_pose.return_value = (
            state.position_m,
            (
                1.0,
                0.0,
                0.0,
                0.0,
            ),
        )
        prim.get_linear_velocity.return_value = (0.0, 0.0, 0.0)
        prims[state.object_id] = prim
    return task, reset, robot, prims, IsaacEpisodeRuntime(robot, prims, reset)


def test_apply_reset_reorders_joints_and_zeros_velocities() -> None:
    _, reset, robot, prims, runtime = setup_runtime()

    runtime.apply_reset()

    robot.set_joint_positions.assert_called_once_with(tuple(reversed(reset.robot_joint_positions_rad)))
    robot.set_joint_velocities.assert_called_once_with((0.0,) * 9)
    for state in reset.objects:
        prims[state.object_id].set_world_pose.assert_called_once()
        prims[state.object_id].set_linear_velocity.assert_called_once_with((0.0, 0.0, 0.0))
        prims[state.object_id].set_angular_velocity.assert_called_once_with((0.0, 0.0, 0.0))


def test_capture_frame_feeds_privileged_evaluator() -> None:
    task, reset, _, _, runtime = setup_runtime()

    result = PrivilegedEpisodeEvaluator(task, reset).update(runtime.capture_frame(0.0))

    assert result.status is EpisodeStatus.RUNNING
    assert not result.checkpoints.target_grasped


def test_runtime_validates_handles_and_time_step() -> None:
    _, reset, robot, prims, runtime = setup_runtime()

    with pytest.raises(ValueError, match="object handle mismatch"):
        IsaacEpisodeRuntime(robot, {"basket": prims["basket"]}, reset)
    with pytest.raises(ValueError, match="positive"):
        runtime.settle_step_count(0.0)
    assert runtime.settle_step_count(1 / 60) == 120
