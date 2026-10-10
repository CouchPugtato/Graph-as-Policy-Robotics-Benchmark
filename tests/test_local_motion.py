from unittest.mock import Mock

import pytest

from gap_isaac.local_motion import LocalTrajectoryExecutor, MotionSafetyPolicy
from gap_isaac.protocol import ExecuteTrajectoryRequest, TrajectoryPoint


class FakeProperties:
    def __init__(self) -> None:
        self._rows = [
            {"lower": -1.0, "upper": 1.0, "maxVelocity": 2.0},
            {"lower": -2.0, "upper": 2.0, "maxVelocity": 2.0},
        ]

    def __getitem__(self, index: int) -> dict[str, float]:
        return self._rows[index]


class FakeRobot:
    dof_names = ("joint1", "joint2")
    dof_properties = FakeProperties()

    def __init__(self) -> None:
        self.positions = [0.0, 0.0]
        self.actions: list[tuple[tuple[float, ...], tuple[int, ...]]] = []

    def get_joint_positions(self) -> list[float]:
        return self.positions

    def apply_action(self, action: tuple[tuple[float, ...], tuple[int, ...]]) -> None:
        targets, indices = action
        self.actions.append(action)
        for target, index in zip(targets, indices, strict=True):
            self.positions[index] = target


def test_optional_velocity_feed_forward_is_bounded_and_zeroed_on_settle():
    robot = FakeRobot()
    velocity_commands = []

    def factory(targets, velocities, indices):
        velocity_commands.append(velocities)
        return targets, indices

    runner = LocalTrajectoryExecutor(
        robot, physics_dt_s=.1, step=lambda: None,
        action_factory=lambda targets, indices: (targets, indices),
        velocity_action_factory=factory,
    )
    result = runner.execute(request((.5, (.1,)), (1.0, (.05,))))
    assert result.status == "succeeded"
    assert any(v[0] > 0 for v in velocity_commands)
    assert any(v[0] < 0 for v in velocity_commands)
    assert max(abs(v[0]) for v in velocity_commands) <= .2 + 1e-12
    assert velocity_commands[-1] == (0.0,)


def test_optional_velocity_feed_forward_is_zeroed_on_safety_stop():
    robot = FakeRobot()
    velocity_commands = []

    def factory(targets, velocities, indices):
        velocity_commands.append(velocities)
        return targets, indices

    runner = LocalTrajectoryExecutor(
        robot, physics_dt_s=.1, step=lambda: None,
        action_factory=lambda targets, indices: (targets, indices),
        velocity_action_factory=factory, safety_check=lambda: "collision",
    )
    result = runner.execute(request((1.0, (.1,))))
    assert result.status == "failed"
    assert velocity_commands[0] == pytest.approx((.1,))
    assert velocity_commands[-1] == (0.0,)


def test_tracking_pause_holds_then_resumes_without_skipping_path():
    robot = FakeRobot()
    decisions = iter((False, False, True))
    commands = []

    def factory(targets, velocities, indices):
        commands.append((targets, velocities))
        return targets, indices

    runner = LocalTrajectoryExecutor(
        robot, physics_dt_s=.1, step=lambda: None,
        action_factory=lambda targets, indices: (targets, indices),
        velocity_action_factory=factory,
        progress_check=lambda: next(decisions, True),
    )
    assert runner.execute(request((1.0, (.1,)))).status == "succeeded"
    assert commands[:2] == [((0.0,), (0.0,))] * 2
    assert robot.positions[0] == pytest.approx(.1)
    assert commands[-1][1] == (0.0,)


def test_tracking_pause_cannot_extend_local_duration_limit():
    robot = FakeRobot()
    commands = []

    def factory(targets, velocities, indices):
        commands.append(velocities)
        return targets, indices

    runner = LocalTrajectoryExecutor(
        robot, physics_dt_s=.1, step=lambda: None,
        action_factory=lambda targets, indices: (targets, indices),
        velocity_action_factory=factory, progress_check=lambda: False,
        policy=MotionSafetyPolicy(max_trajectory_duration_s=.5),
    )
    result = runner.execute(request((.4, (.1,))))
    assert result.status == "timed_out"
    assert "local duration limit" in result.detail
    assert len(commands) <= 7
    assert all(v == (0.0,) for v in commands)


def request(*points: tuple[float, tuple[float, ...]], timeout_s: float = 3.0):
    return ExecuteTrajectoryRequest(
        command_id="local-test",
        joint_names=("joint1",),
        points=tuple(
            TrajectoryPoint(time_from_start_s=time_s, positions_rad=positions)
            for time_s, positions in points
        ),
        timeout_s=timeout_s,
    )


def executor(
    robot: FakeRobot,
    *,
    step=None,
    safety_check=None,
    policy: MotionSafetyPolicy | None = None,
) -> LocalTrajectoryExecutor:
    return LocalTrajectoryExecutor(
        robot,
        physics_dt_s=0.1,
        step=step or (lambda: None),
        action_factory=lambda targets, indices: (targets, indices),
        safety_check=safety_check,
        policy=policy,
        allowed_joint_names=("joint1",),
    )


def test_executes_bounded_trajectory_and_reaches_final_target() -> None:
    robot = FakeRobot()
    result = executor(robot).execute(request((1.0, (0.2,)), (2.0, (0.0,))))

    assert result.status == "succeeded"
    assert robot.positions[0] == 0.0
    assert robot.actions


def test_rejects_limits_speed_duration_and_disabled_joints() -> None:
    robot = FakeRobot()
    local = executor(robot)
    outside_limit = request((1.0, (1.0,)))
    too_fast = request((0.1, (0.2,)))
    too_long = request((6.0, (0.2,)), timeout_s=7.0)
    disabled = ExecuteTrajectoryRequest(
        command_id="disabled",
        joint_names=("joint2",),
        points=(TrajectoryPoint(time_from_start_s=1.0, positions_rad=(0.1,)),),
        timeout_s=2.0,
    )

    assert local.execute(outside_limit).status == "failed"
    assert local.execute(too_fast).status == "failed"
    assert local.execute(too_long).status == "failed"
    assert local.execute(disabled).status == "failed"
    assert not robot.actions


def test_stop_latches_and_holds_current_position() -> None:
    robot = FakeRobot()
    holder: list[LocalTrajectoryExecutor] = []

    def stop_after_first_step() -> None:
        holder[0].request_stop()

    local = executor(robot, step=stop_after_first_step)
    holder.append(local)
    result = local.execute(request((1.0, (0.2,))))

    assert result.status == "canceled"
    assert local.execute(request((1.0, (0.0,)))).status == "canceled"
    assert len(robot.actions) == 2


def test_safety_monitor_aborts_and_holds() -> None:
    robot = FakeRobot()
    check = Mock(return_value="disallowed collision")
    result = executor(robot, safety_check=check).execute(request((1.0, (0.2,))))

    assert result.status == "failed"
    assert result.detail == "disallowed collision"
    assert len(robot.actions) == 2


def test_final_tracking_error_fails() -> None:
    robot = FakeRobot()

    def ignore_action(action) -> None:
        robot.actions.append(action)

    robot.apply_action = ignore_action
    policy = MotionSafetyPolicy(final_position_tolerance=0.001, final_settle_time_s=0.1)
    result = executor(robot, policy=policy).execute(request((1.0, (0.2,))))

    assert result.status == "failed"
    assert "final joint error" in result.detail
    assert "joint=joint1" in result.detail
    assert "error_before_settle=0.200000" in result.detail
    assert robot.actions[-1][0] == (0.0,)


def test_extended_settling_allows_lagging_arm_to_converge() -> None:
    robot = FakeRobot()

    def lagging_action(action) -> None:
        targets, indices = action
        robot.actions.append(action)
        for target, index in zip(targets, indices, strict=True):
            robot.positions[index] += 0.1 * (target - robot.positions[index])

    robot.apply_action = lagging_action
    policy = MotionSafetyPolicy(max_final_settle_time_s=2.0)
    result = executor(robot, policy=policy).execute(request((1.0, (0.2,)), timeout_s=4.0))

    assert result.status == "succeeded"
    assert abs(robot.positions[0] - 0.2) <= policy.final_position_tolerance
    assert 15 < len(robot.actions) < 30


def test_extended_settling_still_rejects_persistent_error() -> None:
    robot = FakeRobot()
    robot.apply_action = lambda action: robot.actions.append(action)
    policy = MotionSafetyPolicy(max_final_settle_time_s=2.0)
    result = executor(robot, policy=policy).execute(request((1.0, (0.2,)), timeout_s=4.0))

    assert result.status == "failed"
    assert "settle_time_s=2.000" in result.detail
    assert "error_before_settle=0.200000" in result.detail


def test_safety_monitor_remains_active_during_extended_settling() -> None:
    robot = FakeRobot()
    robot.apply_action = lambda action: robot.actions.append(action)
    monitored_steps = 0

    def safety_check():
        nonlocal monitored_steps
        monitored_steps += 1
        return "disallowed collision" if monitored_steps == 20 else None

    policy = MotionSafetyPolicy(max_final_settle_time_s=4.0)
    result = executor(robot, policy=policy, safety_check=safety_check).execute(
        request((1.0, (0.2,)), timeout_s=6.0)
    )

    assert result.status == "failed"
    assert result.detail == "disallowed collision"
    assert monitored_steps == 20
    assert robot.actions[-1][0] == (0.0,)


def test_timeout_includes_final_settling() -> None:
    robot = FakeRobot()
    result = executor(robot).execute(request((1.0, (0.2,)), timeout_s=1.0))

    assert result.status == "timed_out"


def test_loaded_motion_preserves_existing_holding_target_at_start() -> None:
    robot = FakeRobot()
    result = executor(robot).execute(request((1.0, (0.2,))), initial_targets=(0.015,))

    assert result.status == "succeeded"
    # First ramp step continues from the old command, not actual position 0.
    assert robot.actions[0][0][0] == pytest.approx(0.015 + 0.1 * (0.2 - 0.015))
    assert robot.actions[0][1] == (0,)
    assert robot.positions[1] == 0.0


@pytest.mark.parametrize("initial", [(0.021,), (float("nan"),), (1.0,), (), (0.01, 0.01)])
def test_loaded_motion_rejects_unsafe_or_incomplete_holding_targets(initial) -> None:
    robot = FakeRobot()
    result = executor(robot).execute(request((1.0, (0.2,))), initial_targets=initial)

    assert result.status == "failed"
    assert not robot.actions


def test_loaded_motion_checks_speed_from_holding_command_and_measured_pose() -> None:
    robot = FakeRobot()
    local = executor(robot)
    # From command -0.015 this would move faster than allowed.
    assert local.execute(request((0.1, (0.04,))), initial_targets=(-0.015,)).status == "failed"
    # From measured 0 this would move faster than allowed even though the
    # existing command 0.015 is closer to the destination.
    assert local.execute(request((0.1, (0.055,))), initial_targets=(0.015,)).status == "failed"
    assert not robot.actions


def test_loaded_motion_retains_collision_abort() -> None:
    robot = FakeRobot()
    result = executor(robot, safety_check=lambda: "lost target hold").execute(
        request((1.0, (0.2,))), initial_targets=(0.015,)
    )

    assert result.status == "failed"
    assert result.detail == "lost target hold"
    assert len(robot.actions) == 2
