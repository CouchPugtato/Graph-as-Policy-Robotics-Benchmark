import math

import pytest

from gap_isaac.grasp_alignment import GraspAlignmentPolicy, align_open_gripper


def test_alignment_compensates_measured_tracking_bias() -> None:
    goal = (0.68, 0.15, 0.80)
    command = goal
    moves = []
    observations = []

    def measure():
        return goal, (command[0] - 0.0016, command[1] - 0.0004, command[2] - 0.0037)

    def move(position):
        nonlocal command
        moves.append(position)
        command = position

    result = align_open_gripper(goal, measure=measure, move=move, observations=observations)
    assert len(moves) == 2
    assert result == pytest.approx((0.6816, 0.1504, 0.8037))
    assert observations[-1]["error_m"] < 0.0005
    for start, end in zip([goal, *moves[:-1]], moves, strict=True):
        assert math.dist(start, end) <= 0.003 + 1e-12


def test_aligned_gripper_does_not_move() -> None:
    moves = []
    align_open_gripper(
        (0, 0, 0),
        measure=lambda: ((0, 0, 0), (0, 0, 0)),
        move=moves.append,
        observations=[],
    )
    assert not moves


def test_alignment_refuses_large_initial_error() -> None:
    moves = []
    with pytest.raises(RuntimeError, match="exceeds correction bound"):
        align_open_gripper(
            (0, 0, 0),
            measure=lambda: ((0, 0, 0), (0.01, 0, 0)),
            move=moves.append,
            observations=[],
        )
    assert not moves


def test_alignment_stops_when_motion_does_not_improve_error() -> None:
    moves = []
    with pytest.raises(RuntimeError, match="after 4 corrections"):
        align_open_gripper(
            (0, 0, 0),
            measure=lambda: ((0, 0, 0), (-0.001, 0, 0)),
            move=moves.append,
            observations=[],
        )
    assert len(moves) == 4


def test_alignment_refuses_accumulated_command_drift() -> None:
    moves = []
    with pytest.raises(RuntimeError, match="cumulative correction bound"):
        align_open_gripper(
            (0, 0, 0),
            measure=lambda: ((0, 0, 0), (-0.004, 0, 0)),
            move=moves.append,
            observations=[],
        )
    assert len(moves) == 2


def test_alignment_propagates_collision_failure_without_retry() -> None:
    observations = []

    def unsafe_move(_position):
        raise RuntimeError("disallowed collision")

    with pytest.raises(RuntimeError, match="disallowed collision"):
        align_open_gripper(
            (0, 0, 0),
            measure=lambda: ((0, 0, 0), (-0.001, 0, 0)),
            move=unsafe_move,
            observations=observations,
        )
    assert len(observations) == 1


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_alignment_rejects_nonfinite_measurements(value) -> None:
    with pytest.raises(ValueError, match="finite Cartesian"):
        align_open_gripper(
            (0, 0, 0),
            measure=lambda: ((0, 0, 0), (value, 0, 0)),
            move=lambda _: None,
            observations=[],
        )


def test_alignment_rejects_invalid_policy() -> None:
    with pytest.raises(ValueError, match="bounds"):
        align_open_gripper(
            (0, 0, 0),
            measure=lambda: ((0, 0, 0), (0, 0, 0)),
            move=lambda _: None,
            observations=[],
            policy=GraspAlignmentPolicy(maximum_step_m=0),
        )
