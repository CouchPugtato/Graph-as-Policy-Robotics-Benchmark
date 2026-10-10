import pytest

from gap_isaac.contact_telemetry import ContactTelemetry
from gap_isaac.evaluate import CollisionEvent
from gap_isaac.local_gripper import GripperSafetyPolicy, LocalGripperExecutor


class FakeRobot:
    dof_names = (
        "panda_joint1",
        "panda_finger_joint1",
        "panda_finger_joint2",
    )

    def __init__(self) -> None:
        self.positions = [0.0, 0.04, 0.04]

    def get_joint_positions(self):
        return self.positions

    def apply_action(self, action) -> None:
        targets, indices = action
        for target, index in zip(targets, indices, strict=True):
            self.positions[index] = target


class LaggingRobot(FakeRobot):
    def apply_action(self, action) -> None:
        targets, indices = action
        for target, index in zip(targets, indices, strict=True):
            self.positions[index] += 0.5 * (target - self.positions[index])


def telemetry(held=None, collisions=()):
    return ContactTelemetry(held, collisions, 0)


def executor(
    robot, capture=lambda: telemetry(), *, policy=None, finger_stiffness=None, controlled_names=None
):
    return LocalGripperExecutor(
        robot,
        physics_dt_s=0.1,
        step=lambda: None,
        action_factory=lambda targets, indices: (targets, indices),
        contact_capture=capture,
        policy=policy or GripperSafetyPolicy(max_finger_speed_m_s=0.04, timeout_s=2.0),
        finger_stiffness_n_m=finger_stiffness,
        controlled_finger_names=controlled_names,
    )


def test_free_space_close_and_open() -> None:
    robot = FakeRobot()
    gripper = executor(robot)

    assert gripper.move("close", open_gripper=False).status == "succeeded"
    assert gripper.separation_m() <= 0.002
    assert gripper.move("open", open_gripper=True).status == "succeeded"
    assert gripper.separation_m() >= 0.078


def test_time_based_ramp_tolerates_controller_tracking_lag() -> None:
    robot = LaggingRobot()
    gripper = executor(robot)

    assert gripper.move("close", open_gripper=False).status == "succeeded"


def test_required_hold_succeeds_only_for_named_object() -> None:
    robot = FakeRobot()
    captures = [telemetry(), telemetry(), telemetry("alphabet_soup")]

    def capture():
        return captures.pop(0) if captures else telemetry("alphabet_soup")

    result = executor(robot, capture).move(
        "grasp",
        open_gripper=False,
        require_held_object_id="alphabet_soup",
    )

    assert result.status == "succeeded"
    assert result.detail == "holding alphabet_soup at stabilized contact"
    assert robot.positions[1] == 0.028
    assert robot.positions[2] == 0.028


def test_required_hold_fails_if_contact_does_not_stabilize() -> None:
    robot = FakeRobot()
    captures = [telemetry("alphabet_soup"), telemetry()]

    def capture():
        return captures.pop(0) if captures else telemetry()

    result = executor(robot, capture).move(
        "grasp",
        open_gripper=False,
        require_held_object_id="alphabet_soup",
    )

    assert result.status == "failed"
    assert result.detail == "lost alphabet_soup while stabilizing verified contact"


def test_force_preload_uses_stiffness_and_caps_each_finger_deflection() -> None:
    robot = FakeRobot()
    policy = GripperSafetyPolicy(
        max_finger_speed_m_s=0.04,
        grasp_force_n=5.0,
        max_grasp_deflection_m=0.0005,
    )
    gripper = executor(
        robot,
        lambda: telemetry("alphabet_soup"),
        policy=policy,
        finger_stiffness=(20_000.0, 4_000.0),
    )
    result = gripper.move("grasp", open_gripper=False, require_held_object_id="alphabet_soup")

    assert result.status == "succeeded"
    # First contact is reported after the first ramp command (0.036 m).
    assert robot.positions[1] == pytest.approx(0.036 - 5.0 / 20_000.0)
    assert robot.positions[2] == pytest.approx(0.036 - 0.0005)
    assert "force-calibrated preload" in result.detail


def test_force_preload_aborts_on_contact_loss() -> None:
    robot = FakeRobot()
    captures = [telemetry("alphabet_soup"), telemetry()]
    gripper = executor(
        robot,
        lambda: captures.pop(0),
        policy=GripperSafetyPolicy(grasp_force_n=5.0),
        finger_stiffness=(20_000.0, 20_000.0),
    )

    result = gripper.move("grasp", open_gripper=False, require_held_object_id="alphabet_soup")

    assert result.status == "failed"
    assert result.detail == "lost alphabet_soup during force-calibrated preload"


def test_force_preload_requires_valid_drive_stiffness() -> None:
    with pytest.raises(ValueError, match="finger stiffnesses"):
        executor(FakeRobot(), policy=GripperSafetyPolicy(grasp_force_n=5.0))

    with pytest.raises(ValueError, match="finger stiffnesses"):
        executor(
            FakeRobot(),
            policy=GripperSafetyPolicy(grasp_force_n=5.0),
            finger_stiffness=(float("nan"), 1.0),
        )


def test_mimic_gripper_commands_only_the_leader_and_reads_both_fingers() -> None:
    class MimicRobot(FakeRobot):
        def apply_action(self, action):
            assert action[1] == (1,)
            super().apply_action(action)
            self.positions[2] = self.positions[1]

    robot = MimicRobot()
    gripper = executor(
        robot,
        lambda: telemetry("alphabet_soup"),
        policy=GripperSafetyPolicy(
            max_finger_speed_m_s=0.04, grasp_force_n=5.0, grasp_force_multiplier=2.0
        ),
        finger_stiffness=(20_000.0,),
        controlled_names=("panda_finger_joint1",),
    )
    result = gripper.move("grasp", open_gripper=False, require_held_object_id="alphabet_soup")
    assert result.status == "succeeded"
    assert robot.positions[1] == pytest.approx(0.036 - 10.0 / 20_000.0)
    assert gripper.separation_m() == pytest.approx(2 * robot.positions[1])
    assert gripper.move("open", open_gripper=True).status == "succeeded"
    assert gripper.separation_m() >= 0.078


def test_close_without_required_contact_fails() -> None:
    robot = FakeRobot()
    result = executor(robot).move(
        "grasp",
        open_gripper=False,
        require_held_object_id="alphabet_soup",
    )

    assert result.status == "failed"
    assert "without verifying" in result.detail


def test_blocked_gripper_timeout_reports_missing_hold_and_pre_stop_target() -> None:
    class BlockedRobot(FakeRobot):
        def apply_action(self, action):
            targets, indices = action
            for target, index in zip(targets, indices, strict=True):
                self.positions[index] = max(0.0355, target)

    robot = BlockedRobot()
    result = executor(robot).move(
        "grasp", open_gripper=False, require_held_object_id="alphabet_soup"
    )
    assert result.status == "timed_out"
    assert "two-finger hold of alphabet_soup was not verified" in result.detail
    assert "active_targets_m=(0.0, 0.0)" in result.detail
    assert "separation_m=0.071000" in result.detail
    assert "did not reach" not in result.detail


def test_disallowed_collision_aborts() -> None:
    robot = FakeRobot()
    collision = CollisionEvent("finger", "table", 2.0, allowed=False)
    result = executor(robot, lambda: telemetry(collisions=(collision,))).move(
        "close", open_gripper=False
    )

    assert result.status == "failed"
    assert "disallowed collision" in result.detail


def test_stop_is_latched() -> None:
    robot = FakeRobot()
    gripper = executor(robot)
    gripper.request_stop()

    assert gripper.move("close", open_gripper=False).status == "canceled"


def test_asset_stiffness_preload_is_ramped_bounded_and_fixed_under_contact() -> None:
    class ContactRobot(FakeRobot):
        def __init__(self):
            super().__init__()
            self.positions[1:] = [0.035, 0.035]
            self.targets = []

        def apply_action(self, action):
            targets, indices = action
            assert indices == (1,)
            self.targets.append(targets[0])
            # A dynamic contact prevents the fingers reaching the drive target.
            self.positions[1] = max(0.035, targets[0])
            self.positions[2] = self.positions[1]

    robot = ContactRobot()
    gripper = executor(
        robot,
        lambda: telemetry("alphabet_soup"),
        policy=GripperSafetyPolicy(
            grasp_force_n=3.5,
            max_grasp_deflection_m=0.0175,
            grasp_force_multiplier=2.0,
            grasp_settle_s=0.5,
        ),
        finger_stiffness=(400.0,),
        controlled_names=("panda_finger_joint1",),
    )
    result = gripper.move("grasp", open_gripper=False, require_held_object_id="alphabet_soup")
    assert result.status == "succeeded"
    for previous, value in zip([0.035, *robot.targets[:-1]], robot.targets, strict=True):
        assert abs(value - previous) <= 0.02 * 0.1 + 1e-12
        assert value >= 0.0
        assert (0.035 - value) * 400 <= 7.0 + 1e-12
    assert robot.targets[-1] == pytest.approx(0.0175)
    assert robot.targets[-5:] == pytest.approx([0.0175] * 5)
    assert gripper.separation_m() == pytest.approx(0.070)


def test_preload_does_not_unload_an_already_loaded_soft_closing_drive() -> None:
    class LoadedContactRobot(FakeRobot):
        def __init__(self):
            super().__init__()
            self.positions[1:] = [0.035, 0.035]
            self.targets = []

        def apply_action(self, action):
            assert action[1] == (1,)
            self.targets.append(action[0][0])
            # Fingers are blocked by the can, but drive targets can keep moving.

    robot = LoadedContactRobot()

    def capture():
        spring_force = 400.0 * (0.035 - robot.targets[-1])
        return telemetry("alphabet_soup" if spring_force >= 1.0 else None)

    gripper = executor(
        robot,
        capture,
        policy=GripperSafetyPolicy(
            grasp_force_n=3.5,
            max_grasp_deflection_m=0.0175,
            grasp_force_multiplier=2.0,
            grasp_settle_s=0.5,
        ),
        finger_stiffness=(400.0,),
        controlled_names=("panda_finger_joint1",),
    )
    result = gripper.move("grasp", open_gripper=False, require_held_object_id="alphabet_soup")
    assert result.status == "succeeded"
    # The second close step establishes load; preload must not reset to 0.035.
    assert robot.targets[:3] == pytest.approx([0.033, 0.031, 0.029])
    assert gripper.preload_diagnostics["incoming_command_positions_m"] == pytest.approx([0.031])
    assert gripper.preload_diagnostics["contact_positions_m"] == [0.035]
    assert gripper.preload_diagnostics["fixed_preload_targets_m"] == pytest.approx([0.0175])
    for start, end in zip(robot.targets[:-1], robot.targets[1:], strict=True):
        assert -0.02 * 0.1 - 1e-12 <= end - start <= 1e-12
    assert robot.targets[-1] == pytest.approx(0.0175)
