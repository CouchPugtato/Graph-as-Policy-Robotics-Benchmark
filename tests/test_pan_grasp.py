import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gap_libero.pan_grasp import (
    PanPosePlanner,
    configure_pan_gripper,
    fit_pan_pose,
    load_pan_template,
    preshape_pan_gripper,
    settle_pan_grasp,
)


def test_pan_template_reads_vertices_without_a_simulator(tmp_path):
    vertices = np.arange(900, dtype="<f4").reshape(-1, 3)
    mesh = tmp_path / "pan.msh"
    mesh.write_bytes(struct.pack("<4i", len(vertices), 0, 0, 0) + vertices.tobytes())
    assert np.array_equal(load_pan_template(mesh), vertices)
    mesh.write_bytes(struct.pack("<4i", 100001, 0, 0, 0))
    with pytest.raises(ValueError, match="vertex budget"):
        load_pan_template(mesh)


def test_pan_pad_torsion_and_drive_gain_do_not_raise_force_limits():
    model = SimpleNamespace(
        actuator_gainprm=np.ones((4, 10)) * 1000,
        actuator_biasprm=np.ones((4, 10)) * -1000,
        actuator_forcerange=np.tile([-20.0, 20.0], (4, 1)),
        geom_names=["gripper0_finger1_pad_collision", "table", "gripper0_finger2_pad_collision"],
        geom_name2id=lambda name: {
            "gripper0_finger1_pad_collision": 1,
            "table": 2,
            "gripper0_finger2_pad_collision": 3,
        }[name],
        geom_condim=np.full(4, 3),
    )
    env = SimpleNamespace(
        sim=SimpleNamespace(model=model),
        robots=[SimpleNamespace(_ref_joint_gripper_actuator_indexes=[1, 3])],
    )
    configure_pan_gripper(env)
    assert model.geom_condim.tolist() == [3, 4, 3, 4]
    assert model.actuator_gainprm[:, 0].tolist() == [1000, 2000, 1000, 2000]
    assert model.actuator_biasprm[:, 1].tolist() == [-1000, -2000, -1000, -2000]
    assert np.array_equal(model.actuator_forcerange, np.tile([-20.0, 20.0], (4, 1)))


def test_pan_registration_uses_observed_cloud_and_static_template():
    scipy = pytest.importorskip("scipy.spatial.transform")
    rng = np.random.default_rng(7)
    vertices = rng.uniform([-0.08, -0.08, 0.005], [0.08, 0.08, 0.025], (1800, 3))
    rotation = scipy.Rotation.from_euler("xyz", [0.02, -0.03, 0.8]).as_matrix()
    position = np.array([0.1, -0.05, 0.903])
    observed = vertices @ rotation.T + position + rng.normal(0, 0.0001, vertices.shape)
    fit = fit_pan_pose(observed, vertices, position[:2], 0.8, 0.9)
    assert fit["body_position_m"] == pytest.approx(position, abs=0.001)
    assert fit["fit_median_error_m"] < 0.001
    assert fit["fit_inlier_fraction"] > 0.95


def test_pan_registration_excludes_the_distal_handle_stove_overlap():
    spatial = pytest.importorskip("scipy.spatial.transform")
    rng = np.random.default_rng(14)
    bowl = rng.uniform([-0.08, -0.08, 0.005], [0.08, 0.08, 0.025], (1800, 3))
    handle = rng.uniform([0.1, -0.01, 0.035], [0.28, 0.01, 0.045], (800, 3))
    template = np.vstack((bowl, handle))
    rotation = spatial.Rotation.from_euler("z", 0.8).as_matrix()
    position = np.array([0.2, -0.1, 0.903])
    observed = template @ rotation.T + position
    stove = rng.uniform([0.22, -0.02, 0.045], [0.27, 0.02, 0.06], (1500, 3)) @ rotation.T + position
    clean = fit_pan_pose(observed, template, position[:2], 0.8, 0.9)
    merged = fit_pan_pose(np.vstack((observed, stove)), template, position[:2], 0.8, 0.9)
    assert merged["body_position_m"] == pytest.approx(clean["body_position_m"], abs=0.0005)
    assert merged["registration_handle_extent_m"] == 0.19
    assert clean["registration_handle_extent_m"] == 0.28


class ClosingFixture:
    def __init__(self, moving_forever=False, contact=True):
        self.steps = 0
        self.guarded_steps = 0
        self.moving_forever = moving_forever
        self.hold_reference = None
        self.hold_rotation = None
        self.plan = SimpleNamespace(
            target_object="pan", actions={"close_gripper": SimpleNamespace(max_steps=80)}
        )
        self.backend = SimpleNamespace(
            env=SimpleNamespace(
                robots=[SimpleNamespace(_ref_joint_gripper_actuator_indexes=[0, 1])],
                sim=SimpleNamespace(
                    data=SimpleNamespace(ctrl=np.zeros(2), actuator_force=np.ones(2) * 10)
                ),
            ),
            grasped=lambda target: contact,
            target_rotation_in_gripper=lambda target: np.eye(3),
        )

    def grasp_point(self):
        return np.array(
            [0.002 * (self.steps if self.moving_forever else min(self.steps, 20)), 0, 0]
        )

    def command(self, require_grasp=False):
        self.steps += 1
        if require_grasp:
            self.guarded_steps += 1
            assert self.hold_reference is not None
            assert self.hold_rotation is not None


def test_pan_close_waits_for_settling_before_locking_the_reference():
    pytest.importorskip("scipy")
    fixture = ClosingFixture()
    result = settle_pan_grasp(fixture)
    assert result.outcome == "succeeded"
    assert fixture.steps >= 32
    assert fixture.guarded_steps == 3
    assert fixture.hold_reference == pytest.approx([0.04, 0, 0])


@pytest.mark.parametrize("moving,contact", [(True, True), (False, False)])
def test_unstable_pan_close_fails_without_a_hold_reference(moving, contact):
    pytest.importorskip("scipy")
    fixture = ClosingFixture(moving, contact)
    result = settle_pan_grasp(fixture)
    assert result.outcome == "failed"
    assert fixture.steps == 80
    assert fixture.hold_reference is None


@pytest.mark.parametrize("unloaded", [True, False])
def test_pan_close_does_not_verify_contact_before_clamping_is_complete(unloaded):
    pytest.importorskip("scipy")
    fixture = ClosingFixture()
    data = fixture.backend.env.sim.data
    if unloaded:
        data.actuator_force[:] = 1
    else:
        data.ctrl[:] = 0.01
    result = settle_pan_grasp(fixture)
    assert result.outcome == "failed"
    assert fixture.hold_reference is None


@pytest.mark.parametrize("offset,clear", [(0, False), (0.3, True)])
def test_pan_planner_rejects_overlapping_stove_collision_boxes(offset, clear):
    planner = PanPosePlanner.__new__(PanPosePlanner)
    planner.forward = lambda q: None
    planner.floor = -0.2
    planner.data = SimpleNamespace(
        geom_xmat=np.eye(3).reshape(1, 9), geom_xpos=np.array([[offset, 0, 0]])
    )
    vertices = np.array(
        [[x, y, z] for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.05, 0.05)]
    )
    planner.shapes = [(0, vertices, np.eye(3), np.eye(3))]
    planner.gripper_geoms = {0}
    planner.boxes = [(np.zeros(3), np.ones(3) * 0.1, np.eye(3), False)]
    assert planner.clear(np.zeros(7)) is clear


def test_pan_planner_prefers_a_perpendicular_jaw_without_reading_target_world_pose():
    pytest.importorskip("scipy")
    planner = PanPosePlanner.__new__(PanPosePlanner)
    planner.current = np.zeros(7)
    planner.floor = 0
    planner.boxes = []
    planner.rejections = {"path": 0}
    planner.solve = lambda position, rotation, seed: np.zeros(7)
    planner.clear = lambda q: True
    planner.backend = SimpleNamespace(
        env=SimpleNamespace(
            sim=SimpleNamespace(
                model=object(), data=SimpleNamespace(get_body_xpos=lambda name: np.zeros(3))
            ),
            robots=[SimpleNamespace(robot_model=SimpleNamespace(root_body="robot"))],
        ),
        target_object=lambda name: SimpleNamespace(contact_geoms=[]),
    )
    result = planner.choose({"body_position_m": [0.4, 0, 0.003], "body_rotation": np.eye(3)})
    assert result["jaw_skew_rad"] == 0
    assert result["wrist_tilt_rad"] == 0
    assert result["model_grasp_offset_m"][0] == 0.15


def test_pan_planner_resolves_geometry_ids_instead_of_enumerating_names(monkeypatch):
    pytest.importorskip("scipy")
    model = SimpleNamespace(
        geom_group=np.array([1, 1, 1, 0]),
        geom_type=np.array([0, 0, 0, 6]),
        geom_size=np.ones((4, 3)) * 0.01,
    )
    monkeypatch.setitem(
        sys.modules, "mujoco", SimpleNamespace(MjData=lambda m: SimpleNamespace(qpos=np.zeros(9)))
    )
    backend = SimpleNamespace(
        env=SimpleNamespace(
            sim=SimpleNamespace(
                model=SimpleNamespace(
                    _model=model,
                    geom_names=["gripper0_hand_collision"],
                    geom_name2id=lambda name: 3,
                    site_name2id=lambda name: 0,
                ),
                data=SimpleNamespace(qpos=np.zeros(9)),
            ),
            robots=[
                SimpleNamespace(
                    _ref_joint_pos_indexes=list(range(7)),
                    _ref_joint_indexes=list(range(7)),
                    _ref_gripper_joint_pos_indexes=[7, 8],
                    gripper=SimpleNamespace(important_sites={"grip_site": "grip"}),
                )
            ],
        )
    )
    planner = PanPosePlanner(backend, 0, [0, 0], 0.05)
    assert planner.gripper_geoms == {3}
    assert [shape[0] for shape in planner.shapes] == [3]


@pytest.mark.parametrize("stuck", [False, True])
def test_pan_preshape_uses_actuator_commands_and_a_finite_budget(stuck):
    data = SimpleNamespace(ctrl=np.array([0.04, -0.04]), qpos=np.array([0.04, -0.04]))
    robot = SimpleNamespace(
        _ref_joint_gripper_actuator_indexes=[0, 1], _ref_gripper_joint_pos_indexes=[0, 1]
    )
    bindings = SimpleNamespace(
        backend=SimpleNamespace(
            env=SimpleNamespace(robots=[robot], sim=SimpleNamespace(data=data))
        ),
        commands=[],
    )

    def command():
        bindings.commands.append(bindings.gripper)
        if not stuck:
            data.ctrl -= bindings.gripper * np.array([0.0002, -0.0002])
            data.qpos = data.ctrl.copy()

    bindings.command = command
    if stuck:
        with pytest.raises(RuntimeError, match="step budget"):
            preshape_pan_gripper(bindings)
        assert len(bindings.commands) == 160
    else:
        preshape_pan_gripper(bindings)
        assert 1 in bindings.commands
        assert bindings.gripper == 0
        assert np.max(np.abs(data.qpos)) <= 0.023
