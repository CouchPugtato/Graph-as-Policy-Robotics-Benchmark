"""Bounded, warm-started Cartesian descent planning; no simulator/model imports."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gap_isaac.protocol import ExecuteTrajectoryRequest, TrajectoryPoint

DOWN_ROTATION = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))


def down_orientation_error(rotation: Sequence[Sequence[float]]) -> float:
    trace = sum(float(rotation[i][j]) * DOWN_ROTATION[i][j]
                for i in range(3) for j in range(3))
    if not math.isfinite(trace):
        raise ValueError("non-finite FK rotation")
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))


@dataclass(frozen=True)
class DescentPlan:
    request: ExecuteTrajectoryRequest
    positions_m: tuple[tuple[float, ...], ...]
    maximum_fk_error_m: float
    maximum_orientation_error_rad: float


def plan_descent(
    *,
    start: Sequence[float],
    destination: Sequence[float],
    initial_joints: Sequence[float],
    joint_names: Sequence[str],
    joint_speed_limits: Sequence[float],
    solve: Callable,
    forward: Callable,
    max_duration_s: float,
    settle_timeout_s: float,
    step_m: float = .005,
    cartesian_speed_m_s: float = .020,
) -> DescentPlan:
    """Prevalidate every IK sample and interpolated joint chord before any action.

    Sampling approximates a Cartesian path; live tracking and collision checks
    remain necessary. No ground-truth object pose enters the planner.
    """
    start, destination = tuple(start), tuple(destination)
    previous = tuple(float(v) for v in initial_joints)
    limits = tuple(float(v) for v in joint_speed_limits)
    if len(start) != 3 or len(destination) != 3:
        raise ValueError("descent requires XYZ positions")
    if not previous or len(previous) != len(joint_names) or len(limits) != len(previous):
        raise ValueError("descent joint dimensions do not match")
    values = (*start, *destination, *previous, *limits, step_m,
              cartesian_speed_m_s, max_duration_s, settle_timeout_s)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("descent inputs must be finite")
    if min(*limits, step_m, cartesian_speed_m_s, max_duration_s) <= 0 or settle_timeout_s < 0:
        raise ValueError("descent bounds must be positive")
    if destination[2] >= start[2]:
        raise ValueError("descent must move downward")
    if math.dist(start[:2], destination[:2]) > .005:
        raise ValueError("descent must begin aligned within 5 mm")
    distance = math.dist(start, destination)
    count = math.ceil(distance / step_m)
    if count > 80:
        raise ValueError("descent exceeds 80-waypoint bound")
    points = []
    positions = []
    elapsed = 0.0
    max_error = 0.0
    max_angle = 0.0
    previous_position = start
    pending = [(tuple(a + (b - a) * index / count
                      for a, b in zip(start, destination, strict=True)), 0)
               for index in range(1, count + 1)]
    attempts = 0
    # reuse the previous joint pose to keep the descent on one solution branch
    while pending:
        position, subdivisions = pending.pop(0)
        index = len(points) + 1
        attempts += 1
        if attempts > 160:
            raise ValueError("descent exceeds 160-IK-attempt bound")
        joints, solved = solve(position, previous)
        joints = tuple(float(v) for v in joints)
        if not solved or len(joints) != len(previous) or not all(map(math.isfinite, joints)):
            raise ValueError(f"descent IK failed at waypoint {index}/{count}")
        deltas = tuple(abs(a - b) for a, b in zip(joints, previous, strict=True))
        if max(deltas) > .15:
            # A smaller task-space step may stay on the warm-started branch.
            # Never accept the rejected solution or relax the joint-jump bound.
            if subdivisions < 3 and len(points) + len(pending) + 2 <= 80:
                midpoint = tuple((a + b) / 2 for a, b in
                                 zip(previous_position, position, strict=True))
                pending[0:0] = [(midpoint, subdivisions + 1),
                                (position, subdivisions + 1)]
                continue
            joint_index = deltas.index(max(deltas))
            raise ValueError(
                f"descent IK branch jump at waypoint {index}: "
                f"joint={joint_names[joint_index]}, delta={max(deltas):.6f} rad, "
                f"step={math.dist(previous_position, position):.6f} m, "
                f"subdivisions={subdivisions}, position={position}; "
                "no continuous solution found within bounded refinement")
        # Check endpoints and interior samples of each executor-interpolated chord.
        for fraction in (0.0, .25, .5, .75, 1.0):
            sample = tuple(a + (b - a) * fraction
                           for a, b in zip(previous, joints, strict=True))
            expected = tuple(a + (b - a) * fraction
                             for a, b in zip(previous_position, position, strict=True))
            actual, rotation = forward(sample)
            actual = tuple(float(v) for v in actual)
            if len(actual) != 3 or not all(map(math.isfinite, actual)):
                raise ValueError("invalid FK position")
            error = math.dist(actual, expected)
            angle = down_orientation_error(rotation)
            max_error, max_angle = max(max_error, error), max(max_angle, angle)
            if error > .002 or angle > .02:
                raise ValueError(f"descent FK path check failed at waypoint {index}/{count}: "
                                 f"position_error={error:.6f} m; orientation_error={angle:.6f} rad")
        # each segment must respect both cartesian speed and joint speed limits
        duration = max(.04, math.dist(previous_position, position) / cartesian_speed_m_s,
                       max(abs(a - b) / limit * 1.10
                           for a, b, limit in zip(previous, joints, limits, strict=True)))
        elapsed += duration
        if elapsed > max_duration_s:
            raise ValueError("Cartesian descent exceeds existing trajectory duration limit")
        points.append(TrajectoryPoint(time_from_start_s=elapsed, positions_rad=joints))
        positions.append(position)
        previous, previous_position = joints, position
    request = ExecuteTrajectoryRequest(
        command_id="graph-descend-cartesian", joint_names=tuple(joint_names),
        points=tuple(points), timeout_s=elapsed + settle_timeout_s + 1.0,
    )
    return DescentPlan(request, tuple(positions), max_error, max_angle)
