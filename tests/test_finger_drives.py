from dataclasses import replace

import pytest

from gap_isaac.finger_drives import FingerDrive, plan_finger_preload, select_finger_actuators

LEADER = FingerDrive(
    "panda_finger_joint1", 7, "/World/Franka/joints/panda_finger_joint1", 20_000.0, 100.0, 40.0
)
FOLLOWER = FingerDrive(
    "panda_finger_joint2",
    8,
    "/World/Franka/joints/panda_finger_joint2",
    0.0,
    0.0,
    0.0,
    LEADER.joint_path,
    -1.0,
    0.0,
)


def test_zero_gain_mimic_follower_is_not_treated_as_a_motor() -> None:
    assert select_finger_actuators((LEADER, FOLLOWER)) == (LEADER,)


def test_independent_position_drives_remain_supported() -> None:
    right = replace(FOLLOWER, stiffness_n_m=10_000.0, mimic_reference=None)
    assert select_finger_actuators((LEADER, right)) == (LEADER, right)


def test_zero_gain_without_a_mimic_constraint_is_rejected() -> None:
    with pytest.raises(ValueError, match="position drive is missing"):
        select_finger_actuators((LEADER, replace(FOLLOWER, mimic_reference=None)))


@pytest.mark.parametrize(
    "change",
    [
        {"mimic_reference": "/unrelated/joint"},
        {"mimic_gearing": 1.0},
        {"mimic_offset_m": 0.01},
        {"stiffness_n_m": 1.0},
    ],
)
def test_invalid_mimic_relationship_is_rejected(change) -> None:
    with pytest.raises(ValueError):
        select_finger_actuators((LEADER, replace(FOLLOWER, **change)))


def test_preload_sizing_for_the_inspected_isaac_61_asset() -> None:
    # Authored franka.usd values: one 400 N/m drive capped at 7.2 N.
    leader = replace(LEADER, stiffness_n_m=400.0, damping_n_s_m=80.0, effort_limit_n=7.2)
    planned = plan_finger_preload(
        (leader, FOLLOWER), force_per_finger_n=3.5, maximum_deflection_m=0.0005
    )
    assert len(planned) == 1
    assert planned[0].stiffness_n_m == pytest.approx(14_000.0)
    assert planned[0].damping_n_s_m == leader.damping_n_s_m
    assert planned[0].effort_limit_n == leader.effort_limit_n


def test_preload_cannot_exceed_the_active_motor_effort_limit() -> None:
    with pytest.raises(ValueError, match="effort limit"):
        plan_finger_preload(
            (replace(LEADER, effort_limit_n=7.2), FOLLOWER),
            force_per_finger_n=5.0,
            maximum_deflection_m=0.0005,
        )


def test_larger_command_offset_retains_original_asset_stiffness_and_force_limit() -> None:
    leader = replace(LEADER, stiffness_n_m=400.0, damping_n_s_m=80.0, effort_limit_n=7.2)
    planned = plan_finger_preload(
        (leader, FOLLOWER), force_per_finger_n=3.5, maximum_deflection_m=0.0175
    )
    assert planned[0].stiffness_n_m == pytest.approx(400.0)
    assert planned[0].damping_n_s_m == 80.0
    assert planned[0].effort_limit_n == 7.2
    assert planned[0].stiffness_n_m * 0.0175 == pytest.approx(7.0)
