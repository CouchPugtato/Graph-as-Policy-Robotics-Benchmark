from test_local_motion import FakeRobot

from gap_isaac.local_motion_v25 import LocalTrajectoryExecutor, MotionSafetyPolicy
from gap_isaac.protocol import ExecuteTrajectoryRequest, TrajectoryPoint


def execute(check, safety=lambda: None):
    robot = FakeRobot()
    executor = LocalTrajectoryExecutor(
        robot, physics_dt_s=.1, step=lambda: None,
        action_factory=lambda targets, indices: (targets, indices),
        final_completion_check=check, safety_check=safety,
        policy=MotionSafetyPolicy(final_settle_time_s=.2, max_final_settle_time_s=.5))
    request = ExecuteTrajectoryRequest(command_id="align", joint_names=("joint1",),
        points=(TrajectoryPoint(time_from_start_s=.5, positions_rad=(.1,)),), timeout_s=1.5)
    return executor.execute(request), robot


def test_readiness_waits_within_original_budget():
    calls = []
    def ready():
        calls.append(True)
        return len(calls) == 3
    result, robot = execute(ready)
    assert result.status == "succeeded"
    assert len(calls) == 3
    assert len(robot.actions) == 9  # Five path steps, four settle steps.


def test_persistent_endpoint_error_fails_without_unbounded_wait():
    result, robot = execute(lambda: False)
    assert result.status == "failed"
    assert "existing settle limit" in result.detail
    assert len(robot.actions) == 11  # Five path, five settle, safety hold.


def test_safety_stop_precedes_endpoint_success():
    result, robot = execute(lambda: True, lambda: "collision")
    assert result.status == "failed"
    assert result.detail == "collision"
    assert len(robot.actions) == 2
