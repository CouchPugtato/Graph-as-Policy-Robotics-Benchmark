import pytest

from gap_isaac.isaac_sim_adapter import _dof_names


def test_dof_names_uses_isaac_61_single_articulation_property() -> None:
    class Robot:
        dof_names = ["panda_joint1", "panda_joint2"]

    assert _dof_names(Robot(), 2) == ("panda_joint1", "panda_joint2")


def test_dof_names_rejects_state_length_mismatch() -> None:
    class Robot:
        dof_names = ["panda_joint1"]

    with pytest.raises(RuntimeError, match="does not match"):
        _dof_names(Robot(), 2)
