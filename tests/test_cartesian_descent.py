import math

import pytest

from gap_isaac.cartesian_descent import DOWN_ROTATION, down_orientation_error, plan_descent
from gap_isaac.local_motion import LocalTrajectoryExecutor, MotionSafetyPolicy


def make_plan(**overrides):
    arguments = {
        "start": (.4, .2, .3), "destination": (.4, .2, .08),
        "initial_joints": (.4, .2, .3), "joint_names": ("x", "y", "z"),
        "joint_speed_limits": (.35, .35, .35),
        "solve": lambda position, warm: (position, True),
        "forward": lambda joints: (joints, DOWN_ROTATION),
        "max_duration_s": 12, "settle_timeout_s": 4,
    }
    arguments.update(overrides)
    return plan_descent(**arguments)


def test_bounded_waypoints_fixed_xy_and_continuous_warm_start():
    calls = []

    def solve(position, warm):
        calls.append((position, warm))
        return position, True

    plan = make_plan(solve=solve)
    assert len(plan.positions_m) == 44
    assert plan.positions_m[-1] == pytest.approx((.4, .2, .08))
    assert plan.request.points[-1].time_from_start_s <= 12
    assert plan.request.points[-1].time_from_start_s == pytest.approx(.22 / .020)
    assert plan.maximum_fk_error_m < 1e-10
    assert plan.maximum_orientation_error_rad == 0
    previous_position = (.4, .2, .3)
    previous_time = 0
    for position, warm in calls:
        assert warm == previous_position
        assert position[:2] == (.4, .2)
        assert math.dist(previous_position, position) <= .005 + 1e-12
        previous_position = position
    for point in plan.request.points:
        assert point.time_from_start_s > previous_time
        previous_time = point.time_from_start_s


@pytest.mark.parametrize("change, message", [
    ({"destination": (.4, .2, .4)}, "downward"),
    ({"destination": (.42, .2, .08)}, "aligned"),
    ({"destination": (.4, .2, -.2)}, "80-waypoint"),
    ({"step_m": 0}, "positive"),
    ({"start": (float("nan"), .2, .3)}, "finite"),
    ({"joint_speed_limits": (0, .35, .35)}, "positive"),
    ({"joint_names": ("x",)}, "dimensions"),
    ({"max_duration_s": .2}, "duration limit"),
    ({"solve": lambda p, w: (p, False)}, "IK failed"),
    ({"solve": lambda p, w: ((p[0] + .3, p[1], p[2]), True)}, "branch jump"),
    ({"forward": lambda q: ((q[0] + .004, q[1], q[2]), DOWN_ROTATION)}, "FK path"),
    ({"forward": lambda q: (q, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))}, "FK path"),
])
def test_rejects_invalid_or_unsafe_paths_before_execution(change, message):
    with pytest.raises(ValueError, match=message):
        make_plan(**change)


def test_fk_checks_interpolated_chords_not_only_endpoints():
    def forward(q):
        # Endpoints every 5 mm are correct; joint interpolation bows out sideways.
        phase = (q[2] - .08) / .005
        bow = .004 * abs(math.sin(math.pi * phase))
        return (q[0] + bow, q[1], q[2]), DOWN_ROTATION

    with pytest.raises(ValueError, match="FK path"):
        make_plan(forward=forward)


def test_branch_jump_refines_step_without_accepting_discontinuous_solution():
    calls = []

    def solve(position, warm):
        calls.append((position, warm))
        if abs(position[2] - warm[2]) > .003:
            return (position[0] + .3, position[1], position[2]), True
        return position, True

    plan = make_plan(destination=(.4, .2, .2), solve=solve)
    assert len(plan.positions_m) == 40
    assert all(point.positions_rad[0] == .4 for point in plan.request.points)
    assert len(calls) == 60


def test_persistent_branch_jump_reports_joint_and_bounded_refinement():
    calls = []

    def solve(position, warm):
        calls.append(position)
        return (position[0] + .3, position[1], position[2]), True

    with pytest.raises(ValueError, match=r"joint=x, delta=0.300000.*subdivisions=3"):
        make_plan(solve=solve)
    assert len(calls) == 4


def test_orientation_check_for_down_pose_and_invalid_rotation():
    assert down_orientation_error(DOWN_ROTATION) == 0
    assert down_orientation_error(((1, 0, 0), (0, 1, 0), (0, 0, 1))) == math.pi
    with pytest.raises(ValueError, match="non-finite"):
        down_orientation_error(((float("nan"), 0, 0), (0, 1, 0), (0, 0, 1)))


class FakeRobot:
    dof_names = ("x", "y", "z")
    dof_properties = [{"lower": -2, "upper": 2, "maxVelocity": 2}] * 3

    def __init__(self):
        self.positions = (.4, .2, .3)
        self.commands = []

    def get_joint_positions(self):
        return self.positions

    def apply_action(self, action):
        self.commands.append(action)
        self.positions = action[0]


def test_existing_executor_enforces_original_limits_and_safety_on_waypoint_path():
    plan = make_plan()
    robot = FakeRobot()
    limits = MotionSafetyPolicy(max_trajectory_duration_s=12, max_command_speed_rad_s=.35,
                                final_position_tolerance=.02, final_settle_time_s=.5,
                                max_final_settle_time_s=4)
    executor = LocalTrajectoryExecutor(
        robot, physics_dt_s=1 / 120, step=lambda: None,
        action_factory=lambda targets, indices: (targets, indices), policy=limits,
    )
    result = executor.execute(plan.request)
    assert result.status == "succeeded"
    assert robot.positions == pytest.approx((.4, .2, .08))
    blocked = LocalTrajectoryExecutor(
        FakeRobot(), physics_dt_s=1 / 120, step=lambda: None,
        action_factory=lambda targets, indices: (targets, indices), policy=limits,
        safety_check=lambda: "disallowed collision",
    )
    result = blocked.execute(plan.request)
    assert result.status == "failed"
    assert "disallowed collision" in result.detail
