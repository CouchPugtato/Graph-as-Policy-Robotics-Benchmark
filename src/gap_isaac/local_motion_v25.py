"""Bounded simulator-local joint motion with hard safety validation."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from gap_isaac.protocol import CommandResult, ExecuteTrajectoryRequest


@dataclass(frozen=True)
class MotionSafetyPolicy:
    max_trajectory_duration_s: float = 5.0
    max_command_speed_rad_s: float = 0.5
    robot_velocity_scale: float = 0.25
    joint_limit_margin: float = 0.005
    final_position_tolerance: float = 0.02
    final_settle_time_s: float = 0.5
    max_final_settle_time_s: float | None = None


class LocalTrajectoryExecutor:
    """Execute validated arm trajectories on Isaac's simulation thread only."""

    def __init__(
        self,
        robot: Any,
        *,
        physics_dt_s: float,
        step: Callable[[], None],
        action_factory: Callable[[tuple[float, ...], tuple[int, ...]], Any],
        safety_check: Callable[[], str | None] | None = None,
        policy: MotionSafetyPolicy | None = None,
        allowed_joint_names: Sequence[str] | None = None,
        velocity_action_factory: Callable[
            [tuple[float, ...], tuple[float, ...], tuple[int, ...]], Any
        ] | None = None,
        progress_check: Callable[[], bool] | None = None,
        final_completion_check: Callable[[], bool] | None = None,
    ) -> None:
        if physics_dt_s <= 0:
            raise ValueError("physics time step must be positive")
        self._robot = robot
        self._physics_dt_s = physics_dt_s
        self._step = step
        self._action_factory = action_factory
        self._velocity_action_factory = velocity_action_factory
        self._progress_check = progress_check
        self._final_completion_check = final_completion_check or (lambda: True)
        self._safety_check = safety_check or (lambda: None)
        self._policy = policy or MotionSafetyPolicy()
        maximum_settle = self._policy.max_final_settle_time_s
        if maximum_settle is not None and maximum_settle < self._policy.final_settle_time_s:
            raise ValueError("maximum final settle time must cover the minimum settle time")
        names = getattr(robot, "dof_names", None)
        if names is None:
            names = getattr(robot, "joint_names", None)
        if names is None:
            raise RuntimeError("robot articulation does not expose DOF names")
        self._joint_names = tuple(str(name) for name in names)
        self._allowed_joint_names = set(allowed_joint_names or self._joint_names)
        if not self._allowed_joint_names <= set(self._joint_names):
            raise ValueError("allowed joints are not present in the robot articulation")
        self._stop = threading.Event()

    def request_stop(self) -> None:
        """Latch a stop request. A new executor is required before further motion."""

        # a stop remains active until a new executor is created
        self._stop.set()

    def _properties(self, indices: tuple[int, ...]) -> tuple[tuple[float, float, float], ...]:
        properties = self._robot.dof_properties
        return tuple(
            (
                float(properties[index]["lower"]),
                float(properties[index]["upper"]),
                float(properties[index]["maxVelocity"]),
            )
            for index in indices
        )

    def _failure(self, request: ExecuteTrajectoryRequest, detail: str) -> CommandResult:
        return CommandResult(command_id=request.command_id, status="failed", detail=detail)

    def _hold(self, indices: tuple[int, ...]) -> None:
        current = tuple(float(self._robot.get_joint_positions()[index]) for index in indices)
        self._command(current, indices)

    def _command(
        self, targets: tuple[float, ...], indices: tuple[int, ...],
        velocities: tuple[float, ...] | None = None,
    ) -> None:
        if self._velocity_action_factory is None:
            action = self._action_factory(targets, indices)
        else:
            action = self._velocity_action_factory(
                targets, velocities or (0.0,) * len(targets), indices)
        self._robot.apply_action(action)

    def _validate(
        self,
        request: ExecuteTrajectoryRequest,
        initial_targets: tuple[float, ...] | None = None,
    ) -> tuple[tuple[int, ...], tuple[float, ...], tuple[tuple[float, float, float], ...]]:
        if len(set(request.joint_names)) != len(request.joint_names):
            raise ValueError("trajectory contains duplicate joint names")
        unknown = set(request.joint_names) - self._allowed_joint_names
        if unknown:
            raise ValueError(f"trajectory contains disabled joints: {sorted(unknown)}")
        indices = tuple(self._joint_names.index(name) for name in request.joint_names)
        current_all = self._robot.get_joint_positions()
        current = tuple(float(current_all[index]) for index in indices)
        properties = self._properties(indices)
        initial = current
        if initial_targets is not None:
            if len(initial_targets) != len(indices):
                raise ValueError("initial holding targets must contain every requested joint")
            for offset, target in enumerate(initial_targets):
                if not math.isfinite(target):
                    raise ValueError("initial holding targets must be finite")
                lower, upper, _ = properties[offset]
                margin = self._policy.joint_limit_margin
                if target < lower + margin or target > upper - margin:
                    raise ValueError("initial holding target is outside safety-margined limits")
                if abs(target - current[offset]) > self._policy.final_position_tolerance:
                    raise ValueError(
                        "initial holding target differs from measured pose beyond tolerance"
                    )
            initial = initial_targets
        final_time = request.points[-1].time_from_start_s
        if final_time > self._policy.max_trajectory_duration_s:
            raise ValueError("trajectory exceeds the local duration limit")
        if final_time > request.timeout_s:
            raise ValueError("trajectory duration exceeds its timeout")

        previous_time = 0.0
        previous_positions = initial
        for point in request.points:
            duration = point.time_from_start_s - previous_time
            if duration <= 0:
                raise ValueError("first trajectory point must occur after time zero")
            for offset, position in enumerate(point.positions_rad):
                if not math.isfinite(position):
                    raise ValueError("trajectory positions must be finite")
                lower, upper, robot_max_velocity = properties[offset]
                margin = self._policy.joint_limit_margin
                if position < lower + margin or position > upper - margin:
                    name = request.joint_names[offset]
                    raise ValueError(f"{name} target is outside its safety-margined limits")
                speed = abs(position - previous_positions[offset]) / duration
                if previous_time == 0.0:
                    # Retain the measured-pose speed check as well as command
                    # continuity; preserving a preload cannot bypass limits.
                    speed = max(speed, abs(position - current[offset]) / duration)
                allowed_speed = min(
                    self._policy.max_command_speed_rad_s,
                    robot_max_velocity * self._policy.robot_velocity_scale,
                )
                if speed > allowed_speed:
                    name = request.joint_names[offset]
                    raise ValueError(f"{name} segment exceeds the local speed limit")
            previous_time = point.time_from_start_s
            previous_positions = point.positions_rad
        return indices, initial, properties

    @staticmethod
    def _interpolate(
        elapsed_s: float,
        request: ExecuteTrajectoryRequest,
        initial: tuple[float, ...],
    ) -> tuple[float, ...]:
        start_time = 0.0
        start_positions = initial
        for point in request.points:
            if elapsed_s <= point.time_from_start_s:
                fraction = (elapsed_s - start_time) / (point.time_from_start_s - start_time)
                return tuple(
                    start + fraction * (end - start)
                    for start, end in zip(start_positions, point.positions_rad, strict=True)
                )
            start_time = point.time_from_start_s
            start_positions = point.positions_rad
        return request.points[-1].positions_rad

    def execute(
        self,
        request: ExecuteTrajectoryRequest,
        *,
        initial_targets: tuple[float, ...] | None = None,
    ) -> CommandResult:
        """Optionally continue from bounded, already-applied holding targets.

        Resetting a position drive to measured angles removes its static load
        compensation. Loaded callers can preserve command continuity without
        increasing the allowed tracking error or commanding non-arm joints.
        """
        if self._stop.is_set():
            return CommandResult(
                command_id=request.command_id,
                status="canceled",
                detail="local motion stop is latched",
            )
        try:
            indices, initial, _ = self._validate(request, initial_targets)
        except (IndexError, ValueError) as error:
            return self._failure(request, str(error))

        elapsed_s = 0.0
        path_time_s = 0.0
        final_time = request.points[-1].time_from_start_s
        previous_target = initial
        while path_time_s < final_time:
            if self._stop.is_set():
                self._hold(indices)
                return CommandResult(
                    command_id=request.command_id,
                    status="canceled",
                    detail="local motion stop requested",
                )
            advance = self._progress_check is None or self._progress_check()
            next_time = (min(path_time_s + self._physics_dt_s, final_time)
                         if advance else path_time_s)
            command_dt = next_time - path_time_s
            path_time_s = next_time
            elapsed_s += self._physics_dt_s
            target = self._interpolate(path_time_s, request, initial)
            # Secant across this timestep also handles waypoint boundaries.
            # It is bounded by the segment speeds validated before execution.
            velocities = (tuple((end - start) / command_dt for start, end in
                                zip(previous_target, target, strict=True))
                          if command_dt > 0 else (0.0,) * len(target))
            self._command(target, indices, velocities)
            previous_target = target
            self._step()
            safety_error = self._safety_check()
            if safety_error:
                self._hold(indices)
                return self._failure(request, safety_error)
            if elapsed_s > request.timeout_s:
                self._hold(indices)
                return CommandResult(
                    command_id=request.command_id,
                    status="timed_out",
                    detail="trajectory timed out",
                )
            if (self._progress_check is not None
                    and elapsed_s > self._policy.max_trajectory_duration_s):
                self._hold(indices)
                return CommandResult(
                    command_id=request.command_id, status="timed_out",
                    detail="tracking-paced trajectory exceeded the local duration limit",
                )

        final_target = request.points[-1].positions_rad

        def joint_errors() -> tuple[float, ...]:
            actual = self._robot.get_joint_positions()
            return tuple(
                abs(float(actual[index]) - target)
                for index, target in zip(indices, final_target, strict=True)
            )

        initial_error = max(joint_errors())
        maximum_settle_s = self._policy.max_final_settle_time_s
        if maximum_settle_s is None:
            maximum_settle_s = self._policy.final_settle_time_s
        minimum_steps = math.ceil(self._policy.final_settle_time_s / self._physics_dt_s)
        settle_steps = math.ceil(maximum_settle_s / self._physics_dt_s)
        for settle_index in range(settle_steps):
            if self._stop.is_set():
                self._hold(indices)
                return CommandResult(command_id=request.command_id, status="canceled")
            self._command(final_target, indices)
            self._step()
            elapsed_s += self._physics_dt_s
            if elapsed_s > request.timeout_s:
                self._hold(indices)
                return CommandResult(
                    command_id=request.command_id,
                    status="timed_out",
                    detail="final settling timed out",
                )
            safety_error = self._safety_check()
            if safety_error:
                self._hold(indices)
                return self._failure(request, safety_error)
            error = max(joint_errors())
            if (settle_index + 1 >= minimum_steps
                    and error <= self._policy.final_position_tolerance
                    and self._final_completion_check()):
                return CommandResult(
                    command_id=request.command_id,
                    status="succeeded",
                    detail=(
                        f"final joint error {error:.6f} rad; "
                        f"error_before_settle={initial_error:.6f}, "
                        f"settle_time_s={(settle_index + 1) * self._physics_dt_s:.3f}"
                    ),
                )

        errors = joint_errors()
        error = max(errors)
        if error > self._policy.final_position_tolerance:
            worst_offset = errors.index(error)
            actual_position = float(self._robot.get_joint_positions()[indices[worst_offset]])
            self._hold(indices)
            return self._failure(
                request,
                f"final joint error {error:.6f} exceeds tolerance "
                f"{self._policy.final_position_tolerance:.6f}; "
                f"joint={request.joint_names[worst_offset]}, "
                f"target={final_target[worst_offset]:.6f}, actual={actual_position:.6f}, "
                f"error_before_settle={initial_error:.6f}, "
                f"settle_time_s={settle_steps * self._physics_dt_s:.3f}",
            )
        if not self._final_completion_check():
            self._hold(indices)
            return self._failure(
                request, "endpoint readiness not reached within existing settle limit")
        return CommandResult(command_id=request.command_id, status="succeeded")
