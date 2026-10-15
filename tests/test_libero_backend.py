import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gap_isaac.graph_policy import GraphPolicy
from gap_isaac.observation_bundle import ObservationBundle
from gap_libero.actions import LiberoGraphActions, RunPlan, run_graph
from gap_libero.backend import LiberoBackend, matrix_pose
from gap_libero.cli import configure
from gap_libero.windows_runtime import PRIVATE_MACROS, prepare_windows_runtime


class FakeEnvironment:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 1.0])
        self.commands = []
        self.closed = False
        self.held = False
        self.success = False
        self.force_done = False
        self.never_grasp = False
        self.frozen = False
        self.target_relative = np.array([0.0, 0.0, -0.03])
        self.target = SimpleNamespace(contact_geoms=["target_geom"], root_body="target_body")
        self.env = SimpleNamespace(objects_dict={"target": self.target},
                                   _check_grasp=lambda *args: self.held)
        self.sim = SimpleNamespace(
            model=SimpleNamespace(geom_name2id=lambda name: {"hand": 1, "table": 2}[name]),
            data=SimpleNamespace(contact=[], ncon=0,
                get_site_xpos=lambda site: self.position.copy(),
                get_site_xmat=lambda site: np.eye(3),
                get_body_xpos=lambda body: self.position + self.target_relative))
        self.robots = [SimpleNamespace(
            gripper=SimpleNamespace(important_sites={"grip_site": "grip"}),
            controller=SimpleNamespace(use_delta=True, input_min=np.full(6, -1),
                input_max=np.ones(6), output_min=np.full(6, -0.05),
                output_max=np.full(6, 0.05)))]

    def obs(self):
        return {"robot0_eef_pos": self.position.copy(), "robot0_eef_quat": [0, 0, 0, 1],
                "robot0_joint_pos": np.zeros(7),
                "agentview_image": np.array([[[255, 0, 0]], [[0, 0, 255]]], dtype=np.uint8),
                "agentview_depth": np.array([[[0.1]], [[0.2]]])}

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return self.obs()

    def set_init_state(self, state):
        self.initial_state = state
        return self.obs()

    def step(self, action):
        self.commands.append(action)
        if not self.frozen:
            self.position += np.asarray(action[:3]) * 0.05
        self.held = action[-1] > 0 and not self.never_grasp
        return self.obs(), 0, self.force_done, {}

    def check_success(self):
        return self.success

    def close(self):
        self.closed = True


def make_backend(**settings):
    env = FakeEnvironment()
    backend = LiberoBackend(env, workspace=[[-0.5, -0.5, 0.5], [0.5, 0.5, 1.5]],
                            **settings)
    backend.reset(2, [1, 2])
    return backend, env


def make_plan(**settings):
    return RunPlan.model_validate({
        "suite": "libero_spatial", "task_id": 0, "target_object": "target",
        "workspace_m": [[-0.5, -0.5, 0.5], [0.5, 0.5, 1.5]],
        "actions": {"attempt": {"kind": "move", "position_m": [0.02, 0, 1]},
                    "close": {"kind": "close", "max_steps": 10},
                    "open": {"kind": "open", "max_steps": 10},
                    "verify": {"kind": "verify"}}, **settings})


def make_policy(action="attempt"):
    return GraphPolicy.from_dict({"schema_version": 1, "graph_id": "libero-test",
        "start": "test", "terminals": {"DONE": "succeeded", "FAILED": "failed"},
        "nodes": [{"id": "test", "skill": "test", "action": action,
                   "routes": {"succeeded": "DONE", "failed": "FAILED"}}]})


def test_reset_uses_explicit_initial_state_and_close_is_idempotent():
    backend, env = make_backend()
    assert env.seed_value == 2 and env.initial_state == [1, 2]
    backend.close()
    backend.close()
    assert env.closed
    with pytest.raises(RuntimeError, match="closed"):
        backend.reset(0)


def test_motion_is_interlocked_and_budgeted():
    backend, env = make_backend(max_steps=1)
    with pytest.raises(RuntimeError, match="disabled"):
        backend.step([0] * 7)
    backend.motion_enabled = True
    backend.step([0] * 7)
    with pytest.raises(RuntimeError, match="budget"):
        backend.step([0] * 7)
    assert len(env.commands) == 1


@pytest.mark.parametrize("action", [[0] * 6, [float("nan")] * 7, [2] * 7])
def test_invalid_action_never_reaches_simulator(action):
    backend, env = make_backend(motion_enabled=True)
    with pytest.raises(ValueError):
        backend.step(action)
    assert not env.commands


def test_workspace_and_forbidden_contacts_fail_closed():
    backend, env = make_backend(motion_enabled=True, forbidden_geom_pairs=[("hand", "table")])
    env.sim.data.contact = [SimpleNamespace(geom1=2, geom2=1)]
    env.sim.data.ncon = 1
    with pytest.raises(RuntimeError, match="contact"):
        backend.step([0] * 7)
    env.sim.data.ncon = 0
    backend.observation["robot0_eef_pos"] = [2, 0, 1]
    with pytest.raises(RuntimeError, match="workspace"):
        backend.step([0] * 7)
    assert not env.commands


def test_capture_flips_rgb_and_metric_depth_together_and_round_trips(tmp_path):
    backend, _ = make_backend()
    helpers = SimpleNamespace(
        get_real_depth_map=lambda sim, raw: raw + 1,
        get_camera_intrinsic_matrix=lambda *args: np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]),
        get_camera_extrinsic_matrix=lambda *args: np.eye(4))
    bundle = backend.capture(helpers)
    assert bundle.rgb[:3] == bytes([0, 0, 255])
    assert np.frombuffer(bundle.depth, dtype="<f4").tolist() == pytest.approx([1.2, 1.1])
    bundle.save(tmp_path / "capture")
    loaded = ObservationBundle.load(tmp_path / "capture")
    assert loaded.source == "libero-rgbd-only"
    assert loaded.pixel_to_world(0, 0) == pytest.approx([0, 0, 1.2])
    with pytest.raises(FileExistsError):
        bundle.save(tmp_path / "capture")


@pytest.mark.parametrize("rotation", [np.eye(3), np.diag([1, -1, -1]),
                                     np.diag([-1, 1, -1]), np.diag([-1, -1, 1])])
def test_camera_quaternion_handles_half_turns(rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    q = matrix_pose(matrix).orientation
    assert q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w == pytest.approx(1)


def test_rejects_improper_camera_rotation():
    matrix = np.eye(4)
    matrix[0, 0] = -1
    with pytest.raises(ValueError, match="proper"):
        matrix_pose(matrix)


def test_cartesian_action_is_bounded_and_closing_verifies_grasp():
    backend, env = make_backend(motion_enabled=True)
    actions = LiberoGraphActions(backend, make_plan())
    result = actions.execute_action("attempt", {})
    assert result.outcome == "succeeded"
    assert all(max(abs(value) for value in command[:3]) <= 0.15 for command in env.commands)
    assert actions.execute_action("close", {}).outcome == "succeeded"
    assert actions.hold_reference is not None
    assert actions.execute_action("open", {}).outcome == "succeeded"
    assert actions.hold_reference is None


def test_contact_and_slip_guards():
    backend, env = make_backend(motion_enabled=True)
    actions = LiberoGraphActions(backend, make_plan())
    actions.execute_action("close", {})
    env.target_relative += [0, 0, -0.02]
    with pytest.raises(RuntimeError, match="slipped"):
        actions.check_hold()
    env.target_relative += [0, 0, 0.02]
    env.held = False
    actions.check_hold()
    actions.check_hold()
    with pytest.raises(RuntimeError, match="contact was lost"):
        actions.check_hold()


def test_failed_grasp_and_motion_timeout_are_real_failures():
    backend, env = make_backend(motion_enabled=True)
    actions = LiberoGraphActions(backend, make_plan())
    env.never_grasp = True
    assert actions.execute_action("close", {}).outcome == "failed"
    env.frozen = True
    assert actions.execute_action("attempt", {}).outcome == "failed"


def test_graph_terminal_alone_does_not_prove_task_success():
    backend, _ = make_backend(motion_enabled=True)
    result = run_graph(backend, make_policy(), make_plan())
    assert result["graph"]["status"] == "succeeded"
    assert result["ok"] is False
    assert result["evaluation"]["task_success"] is False


def test_graph_result_routes_verified_task_success():
    backend, env = make_backend(motion_enabled=True)
    original = env.step

    def complete(action):
        result = original(action)
        env.success = True
        return result[0], result[1], True, result[3]

    env.step = complete
    result = run_graph(backend, make_policy(), make_plan())
    assert result["ok"] and result["steps"] == 1


def test_initial_success_and_disabled_motion_are_rejected():
    backend, env = make_backend()
    with pytest.raises(RuntimeError, match="enable motion"):
        run_graph(backend, make_policy(), make_plan())
    backend.motion_enabled = True
    env.success = True
    with pytest.raises(RuntimeError, match="already successful"):
        run_graph(backend, make_policy(), make_plan())


def test_plan_rejects_unbounded_positions_and_unknown_fields():
    with pytest.raises(ValueError, match="outside"):
        make_plan(actions={"attempt": {"kind": "move", "position_m": [2, 0, 1]}})
    with pytest.raises(ValueError):
        make_plan(max_episode_steps=1001)
    with pytest.raises(ValueError):
        make_plan(actions={"attempt": {"kind": "hold", "position_m": [0, 0, 1]}})


def test_metadata_diagnostic_does_not_import_simulators():
    process = subprocess.run([sys.executable, "-c",
        "import sys; from gap_libero.cli import check; check(); "
        "assert not any(n in sys.modules for n in ['libero', 'robosuite', 'mujoco', 'torch'])"],
        capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr


def test_configuration_is_noninteractive_and_does_not_overwrite(tmp_path: Path):
    source = tmp_path / "source"
    for name in ("bddl_files", "init_files", "assets"):
        (source / "libero" / "libero" / name).mkdir(parents=True)
    result = configure(source, tmp_path / "config")
    assert result["ok"] and not result["simulator_started"]
    with pytest.raises(FileExistsError):
        configure(source, tmp_path / "config")


def test_cli_failure_saves_receipt_without_starting_simulator(tmp_path):
    output = tmp_path / "result"
    process = subprocess.run([sys.executable, "-m", "gap_libero.cli", "run",
        "--graph", "missing.json", "--plan", "missing.json", "--output", str(output)],
        capture_output=True, text=True, timeout=10)
    assert process.returncode == 1
    receipt = json.loads((output / "receipt.json").read_text())
    assert not receipt["ok"] and "enable-motion" in receipt["detail"]
    assert "ValueError" in receipt["traceback"]


def test_setup_separates_torch_wheel_index_from_regular_dependencies():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/windows/prepare-libero.ps1").read_text()
    assert "pip install --upgrade pip --index-url https://pypi.org/simple" in script
    assert "libero/requirements-torch-cpu.txt" in script
    assert "pip install --no-deps 'torch==2.5.1+cpu'" in script
    assert "-m pip check" in script
    assert "--config-settings editable_mode=compat" in script
    assert "gap_libero.cli prepare-runtime" in script
    runtime_requirements = (root / "libero/requirements-runtime.txt").read_text()
    assert "h5py==3.12.1" in runtime_requirements


def test_powershell_defaults_capture_and_keeps_motion_opt_in():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/windows/run-libero.ps1").read_text()
    installer = (root / "scripts/windows/prepare-libero.ps1").read_text()
    assert "[string]$Mode = 'capture'" in runner
    assert "IsaacStopped" not in runner and "ModelsStopped" not in runner
    assert "-not $EnableMotion -or -not $Graph -or -not $Plan" in runner
    assert "PSBoundParameters.ContainsKey('Mode')" in runner
    assert "$Install" not in installer


def test_grocery_bindings_cover_original_graph_actions_without_isaac_waypoints():
    from gap_libero.grocery_comparison import build_action_plan

    backend, env = make_backend(motion_enabled=True)
    backend.workspace = np.array([[-0.35, -0.85, 0.02], [1, 0.85, 1.1]])
    env.env.sim = env.sim
    env.sim.data.get_body_xmat = lambda name: np.eye(3)
    env.env.objects_dict["basket"] = SimpleNamespace(root_body="basket_body")
    env.sim.data.get_body_xpos = lambda name: np.array(
        [0.6, 0.15, 0.055] if name == "target_body" else [0.4, -0.25, 0])
    env.env.objects_dict["alphabet_soup"] = env.target
    plan = build_action_plan(backend, 3)
    assert list(plan.actions) == ["open_gripper", "align_over_target", "descend_to_grasp",
        "close_gripper", "lift_grasped", "move_over_basket", "position_for_release",
        "release_object", "retract_from_basket", "settle_release"]
    assert plan.actions["descend_to_grasp"].position_m == pytest.approx([0.6, 0.15, 0.09])
    assert plan.actions["move_over_basket"].position_m[:2] == pytest.approx([0.4, -0.25])
    assert plan.actions["lift_grasped"].require_grasp
    assert plan.actions["settle_release"].kind == "verify"


def test_orientation_control_uses_grip_site_not_wrist_body_frame():
    backend, _ = make_backend()
    backend.observation["robot0_eef_quat"] = [1, 0, 0, 0]
    actions = LiberoGraphActions(backend, make_plan())
    assert backend.eef_orientation() == pytest.approx([0, 0, 0, 1])
    goal = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    assert actions.rotation_error(goal) == pytest.approx([0, 0, np.pi / 2])
    assert actions.rotation_error(-goal) == pytest.approx([0, 0, np.pi / 2])


def test_invalid_move_orientation_and_preload_are_rejected():
    with pytest.raises(ValueError, match="unit quaternion"):
        make_plan(actions={"attempt": {"kind": "move", "position_m": [0, 0, 1],
                                       "orientation_xyzw": [0, 0, 0, 0]}})
    with pytest.raises(ValueError, match="preload"):
        make_plan(actions={"attempt": {"kind": "hold", "preload_steps": 3}})


@pytest.mark.parametrize("degrees", [0, 30, 89, 90, 120, -170, 180, 350])
def test_parallel_jaw_orientation_preserves_alignment_without_long_wrist_rotation(degrees):
    from gap_libero.grocery_comparison import closest_parallel_jaw_yaw

    yaw = np.deg2rad(degrees)
    reference = np.deg2rad(20)
    result = closest_parallel_jaw_yaw(yaw, reference)
    assert abs(result - reference) <= np.pi / 2 + 1e-12
    assert np.cos(2 * result) == pytest.approx(np.cos(2 * yaw))
    assert np.sin(2 * result) == pytest.approx(np.sin(2 * yaw))


def test_parallel_jaw_orientation_rejects_nonfinite_angles():
    from gap_libero.grocery_comparison import closest_parallel_jaw_yaw

    with pytest.raises(ValueError, match="finite"):
        closest_parallel_jaw_yaw(float("nan"))


def test_parallel_jaw_choice_accounts_for_basket_side_of_robot():
    from gap_libero.grocery_comparison import closest_parallel_jaw_yaw

    result = closest_parallel_jaw_yaw(np.deg2rad(99), np.deg2rad(31))
    assert np.rad2deg(result) == pytest.approx(99)
    result = closest_parallel_jaw_yaw(np.deg2rad(-175), np.deg2rad(-37))
    assert np.rad2deg(result) == pytest.approx(5)


def test_gripper_commands_hold_the_prior_pose_instead_of_following_drift():
    backend, env = make_backend(motion_enabled=True)
    actions = LiberoGraphActions(backend, make_plan())
    original_goal = actions.hold_position.copy()
    env.position[0] += 0.01
    backend.observation = env.obs()
    actions.command()
    assert env.commands[-1][0] < 0
    assert np.linalg.norm(env.position - original_goal) < 0.01
    assert actions.hold_position == pytest.approx(original_goal)


def test_windows_runtime_preparation_is_isolated_and_idempotent(tmp_path, monkeypatch):
    import os

    import gap_libero.windows_runtime as runtime

    if os.name != "nt":
        pytest.skip("Windows runtime preparation")
    environment = tmp_path / ".venv-libero"
    packages = environment / "Lib" / "site-packages"
    (packages / "mujoco").mkdir(parents=True)
    (packages / "robosuite" / "utils").mkdir(parents=True)
    (packages / "mujoco" / "mujoco.dll").write_bytes(b"test dll bytes")
    monkeypatch.setattr(runtime.sys, "prefix", str(environment))
    monkeypatch.setattr(runtime.importlib.metadata, "distribution", lambda name: SimpleNamespace(
        version="1.4.1" if name == "robosuite" else "2.3.7",
        locate_file=lambda child: packages / child))
    first = prepare_windows_runtime()
    assert prepare_windows_runtime() == first
    private = packages / "robosuite" / "macros_private.py"
    assert private.read_text() == PRIVATE_MACROS
    private.write_text("existing user settings")
    with pytest.raises(RuntimeError, match="not overwritten"):
        prepare_windows_runtime()
    assert private.read_text() == "existing user settings"


def test_windows_runtime_preparation_rejects_other_environments(monkeypatch):
    import os

    import gap_libero.windows_runtime as runtime

    if os.name != "nt":
        pytest.skip("Windows runtime preparation")
    monkeypatch.setattr(runtime.sys, "prefix", str(Path(sys.prefix).parent / ".venv-perception"))
    with pytest.raises(RuntimeError, match="isolated"):
        prepare_windows_runtime()
