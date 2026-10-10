"""Deterministic episode reset plans for the grocery task."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from gap_isaac.config import TaskConfig
from gap_isaac.grocery_scene import GroceryScenePlan, build_grocery_scene_plan

FRANKA_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)
FRANKA_HOME_POSITIONS_RAD = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04)


@dataclass(frozen=True)
class ResetObjectState:
    object_id: str
    position_m: tuple[float, float, float]
    yaw_deg: float
    linear_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class EpisodeResetPlan:
    seed: int
    target_id: str
    container_id: str
    robot_joint_names: tuple[str, ...]
    robot_joint_positions_rad: tuple[float, ...]
    objects: tuple[ResetObjectState, ...]
    settle_time_s: float
    timeout_s: float

    def as_dict(self) -> dict:
        return asdict(self)


def reset_plan_from_scene(task: TaskConfig, scene: GroceryScenePlan) -> EpisodeResetPlan:
    """Create the complete state needed to reconstruct the beginning of an episode."""

    # start every episode from the same robot pose and resting object state
    return EpisodeResetPlan(
        seed=scene.seed,
        target_id=scene.target_id,
        container_id=task.container_id,
        robot_joint_names=FRANKA_JOINT_NAMES,
        robot_joint_positions_rad=FRANKA_HOME_POSITIONS_RAD,
        objects=tuple(
            ResetObjectState(
                object_id=item.object_id,
                position_m=item.position_m,
                yaw_deg=item.yaw_deg,
            )
            for item in scene.objects
        ),
        settle_time_s=task.settle_time_s,
        timeout_s=task.timeout_s,
    )


def build_episode_reset_plan(
    task: TaskConfig,
    seed: int,
    target_id: str | None = None,
) -> EpisodeResetPlan:
    """Sample a scene and produce a deterministic reset plan for it."""

    return reset_plan_from_scene(
        task,
        build_grocery_scene_plan(task, seed=seed, target_id=target_id),
    )
