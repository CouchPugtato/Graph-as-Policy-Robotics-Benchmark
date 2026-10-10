"""Small, measured Cartesian corrections while the gripper is still open."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class GraspAlignmentPolicy:
    tolerance_m: float = 0.0005
    maximum_step_m: float = 0.003
    maximum_command_offset_m: float = 0.008
    maximum_corrections: int = 4


def _vector(values: Vector3) -> Vector3:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError("alignment requires three finite Cartesian coordinates")
    return result


def align_open_gripper(
    initial_command: Vector3,
    *,
    measure: Callable[[], tuple[Vector3, Vector3]],
    move: Callable[[Vector3], None],
    observations: list[dict],
    policy: GraspAlignmentPolicy | None = None,
) -> Vector3:
    """Compensate small tracking errors; never declare grasp/contact success.

    measure returns (desired live grasp position, measured TCP position).
    Callbacks must settle and perform normal collision checks. The command
    offset is accumulated to compensate bias, but cannot drift arbitrarily.
    No object pose, finger drive, or contact classification is changed here.
    """
    policy = policy or GraspAlignmentPolicy()
    if (
        not all(
            math.isfinite(value) and value > 0
            for value in (
                policy.tolerance_m,
                policy.maximum_step_m,
                policy.maximum_command_offset_m,
            )
        )
        or policy.maximum_corrections < 0
    ):
        raise ValueError("alignment bounds must be finite and positive")
    initial = _vector(initial_command)
    command = initial
    # correct measured tracking error without allowing an unbounded pose offset
    for iteration in range(policy.maximum_corrections + 1):
        desired, actual = (_vector(value) for value in measure())
        error = tuple(goal - value for goal, value in zip(desired, actual, strict=True))
        error_m = math.sqrt(math.fsum(value * value for value in error))
        observations.append(
            {
                "iteration": iteration,
                "command_position_m": list(command),
                "desired_position_m": list(desired),
                "measured_position_m": list(actual),
                "error_m": error_m,
            }
        )
        if error_m > policy.maximum_command_offset_m:
            raise RuntimeError(f"grasp alignment error {error_m:.6f} m exceeds correction bound")
        if error_m <= policy.tolerance_m:
            return command
        if iteration == policy.maximum_corrections:
            break
        scale = min(1.0, policy.maximum_step_m / error_m)
        proposed = tuple(
            value + correction * scale for value, correction in zip(command, error, strict=True)
        )
        offset_m = math.sqrt(
            math.fsum((value - start) ** 2 for value, start in zip(proposed, initial, strict=True))
        )
        if offset_m > policy.maximum_command_offset_m:
            raise RuntimeError("grasp alignment command exceeds cumulative correction bound")
        move(proposed)
        command = proposed
    raise RuntimeError(
        f"grasp alignment did not reach {policy.tolerance_m:.6f} m tolerance; "
        f"final error={error_m:.6f} m after {policy.maximum_corrections} corrections"
    )
