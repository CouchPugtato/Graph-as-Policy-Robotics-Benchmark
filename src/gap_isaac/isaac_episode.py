"""Isaac-handle adapter for reset application and privileged state capture.

The module intentionally imports no Isaac packages. A native Isaac launcher
creates initialized prim handles and passes them into this adapter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from gap_isaac.evaluate import CollisionEvent, EpisodeFrame, ObjectState
from gap_isaac.reset import EpisodeResetPlan


def _yaw_quaternion_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    half_angle = math.radians(yaw_deg) / 2
    return (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))


def _ordered_joint_positions(robot: Any, reset: EpisodeResetPlan) -> tuple[float, ...]:
    names = getattr(robot, "dof_names", None)
    if names is None:
        names = getattr(robot, "joint_names", None)
    if names is None:
        raise RuntimeError("robot articulation does not expose DOF names")
    configured = dict(zip(reset.robot_joint_names, reset.robot_joint_positions_rad, strict=True))
    actual = tuple(str(name) for name in names)
    if set(actual) != set(configured):
        missing = sorted(set(configured) - set(actual))
        unexpected = sorted(set(actual) - set(configured))
        raise RuntimeError(f"robot DOF mismatch: missing={missing}, unexpected={unexpected}")
    return tuple(configured[name] for name in actual)


class IsaacEpisodeRuntime:
    """Apply resets and capture evaluator-only state from initialized Isaac handles."""

    def __init__(
        self,
        robot: Any,
        object_prims: Mapping[str, Any],
        reset: EpisodeResetPlan,
    ) -> None:
        expected = {item.object_id for item in reset.objects}
        actual = set(object_prims)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(f"object handle mismatch: missing={missing}, unexpected={unexpected}")
        self._robot = robot
        self._object_prims = dict(object_prims)
        self._reset = reset

    def apply_reset(self) -> None:
        """Teleport the robot and task objects to the configured initial state."""

        joint_positions = _ordered_joint_positions(self._robot, self._reset)
        self._robot.set_joint_positions(joint_positions)
        self._robot.set_joint_velocities(tuple(0.0 for _ in joint_positions))

        for state in self._reset.objects:
            prim = self._object_prims[state.object_id]
            prim.set_world_pose(
                position=state.position_m,
                orientation=_yaw_quaternion_wxyz(state.yaw_deg),
            )
            if hasattr(prim, "set_linear_velocity"):
                prim.set_linear_velocity(state.linear_velocity_m_s)
            if hasattr(prim, "set_angular_velocity"):
                prim.set_angular_velocity(state.angular_velocity_rad_s)

    def capture_frame(
        self,
        elapsed_s: float,
        *,
        held_object_id: str | None = None,
        collisions: tuple[CollisionEvent, ...] = (),
        graph_error: str | None = None,
    ) -> EpisodeFrame:
        """Read privileged object kinematics for the evaluator."""

        objects: list[ObjectState] = []
        for reset_state in self._reset.objects:
            prim = self._object_prims[reset_state.object_id]
            position, orientation = prim.get_world_pose()
            yaw_deg = math.degrees(
                2
                * math.atan2(
                    float(orientation[3]),
                    float(orientation[0]),
                )
            )
            velocity = (
                prim.get_linear_velocity()
                if hasattr(prim, "get_linear_velocity")
                else (0.0, 0.0, 0.0)
            )
            objects.append(
                ObjectState(
                    object_id=reset_state.object_id,
                    position_m=tuple(float(value) for value in position),
                    yaw_deg=yaw_deg,
                    linear_velocity_m_s=tuple(float(value) for value in velocity),
                )
            )
        return EpisodeFrame(
            elapsed_s=elapsed_s,
            objects=tuple(objects),
            held_object_id=held_object_id,
            collisions=collisions,
            graph_error=graph_error,
        )

    def settle_step_count(self, physics_dt_s: float) -> int:
        if physics_dt_s <= 0:
            raise ValueError("physics time step must be positive")
        return math.ceil(self._reset.settle_time_s / physics_dt_s)
