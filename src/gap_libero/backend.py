"""Bounded LIBERO adapter with no simulator imports until explicitly created."""

from __future__ import annotations

import hashlib
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gap_isaac.observation_bundle import ObservationBundle
from gap_isaac.protocol import (
    CameraIntrinsics,
    JointState,
    ObservationDescriptor,
    Pose,
    Quaternion,
    Vector3,
)

LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def vector(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} requires {length} finite values")
    return result


def matrix_pose(matrix: Any) -> Pose:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("camera transform requires a finite four by four matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("camera transform rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError("camera transform rotation is not proper")
    if not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("camera transform is not homogeneous")
    # choose the largest quaternion component to handle rotations near pi
    candidates = np.array([
        1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2],
        1 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2],
        1 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2],
        1 + np.trace(rotation),
    ])
    index = int(np.argmax(candidates))
    component = math.sqrt(max(0, float(candidates[index]))) / 2
    denominator = 4 * component
    if index == 3:
        quaternion = [
            (rotation[2, 1] - rotation[1, 2]) / denominator,
            (rotation[0, 2] - rotation[2, 0]) / denominator,
            (rotation[1, 0] - rotation[0, 1]) / denominator, component,
        ]
    else:
        first, second = (index + 1) % 3, (index + 2) % 3
        quaternion = np.zeros(4)
        quaternion[index] = component
        quaternion[first] = (rotation[first, index] + rotation[index, first]) / denominator
        quaternion[second] = (rotation[second, index] + rotation[index, second]) / denominator
        quaternion[3] = (rotation[second, first] - rotation[first, second]) / denominator
    quaternion = vector(quaternion, 4, "quaternion")
    quaternion /= np.linalg.norm(quaternion)
    return Pose(position=Vector3(x=matrix[0, 3], y=matrix[1, 3], z=matrix[2, 3]),
                orientation=Quaternion(x=quaternion[0], y=quaternion[1],
                                       z=quaternion[2], w=quaternion[3]), frame_id="world")


class LiberoBackend:
    def __init__(self, env: Any, *, camera: str = "agentview", motion_enabled: bool = False,
                 max_steps: int = 400, workspace: Any = None,
                 forbidden_geom_pairs: Any = ()) -> None:
        if type(max_steps) is not int or not 1 <= max_steps <= 1000:
            raise ValueError("max_steps must be an integer from one to one thousand")
        self.workspace = np.asarray(workspace, dtype=float) if workspace is not None else None
        if self.workspace is not None and (
                self.workspace.shape != (2, 3) or not np.isfinite(self.workspace).all()
                or not (self.workspace[0] < self.workspace[1]).all()):
            raise ValueError("workspace requires ordered finite minimum and maximum corners")
        self.env, self.camera = env, camera
        self.motion_enabled, self.max_steps = motion_enabled, max_steps
        self.steps, self.closed, self.done = 0, False, False
        self.recorder = None
        self.disallowed_collisions = []
        self.observation: dict[str, Any] | None = None
        self.forbidden_pairs = set()
        for pair in forbidden_geom_pairs:
            if len(pair) != 2 or not all(isinstance(name, str) and name for name in pair):
                raise ValueError("forbidden contact pairs require two geometry names")
            self.forbidden_pairs.add(frozenset(env.sim.model.geom_name2id(name) for name in pair))

    def reset(self, seed: int, init_state: Any = None) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("environment is closed")
        self.env.seed(seed)
        self.observation = self.env.reset()
        if init_state is not None:
            self.observation = self.env.set_init_state(init_state)
        self.steps, self.done = 0, False
        self.disallowed_collisions = []
        self.check_workspace(self.eef_position())
        self.check_contacts()
        return self.observation

    def eef_position(self) -> np.ndarray:
        if self.observation is None:
            raise RuntimeError("reset the environment before reading state")
        return vector(self.observation["robot0_eef_pos"], 3, "end effector position")

    def check_workspace(self, position: Any) -> None:
        position = vector(position, 3, "position")
        if self.workspace is not None and (
                (position < self.workspace[0]).any() or (position > self.workspace[1]).any()):
            raise RuntimeError("end effector is outside the configured workspace")

    def eef_orientation(self) -> np.ndarray:
        site = self.env.robots[0].gripper.important_sites["grip_site"]
        transform = np.eye(4)
        transform[:3, :3] = self.env.sim.data.get_site_xmat(site)
        q = matrix_pose(transform).orientation
        return np.array([q.x, q.y, q.z, q.w])

    def check_contacts(self) -> None:
        if self.forbidden_pairs:
            data = self.env.sim.data
            for contact in data.contact[:data.ncon]:
                if frozenset((contact.geom1, contact.geom2)) in self.forbidden_pairs:
                    name = getattr(self.env.sim.model, "geom_id2name", str)
                    pair = [name(contact.geom1), name(contact.geom2)]
                    self.disallowed_collisions.append({"geometries": pair, "step": self.steps})
                    raise RuntimeError(f"forbidden geometry contact: {pair[0]} with {pair[1]}")

    def step(self, action: Any) -> dict[str, Any]:
        if not self.motion_enabled:
            raise RuntimeError("LIBERO motion is disabled")
        if self.closed or self.done or self.observation is None:
            raise RuntimeError("environment is not ready for motion")
        if self.workspace is None:
            raise RuntimeError("motion requires an explicit workspace")
        if self.steps >= self.max_steps:
            raise RuntimeError("episode step budget exceeded")
        command = vector(action, 7, "OSC pose action")
        if (np.abs(command) > 1).any():
            raise ValueError("OSC pose action exceeds normalized limits")
        self.check_workspace(self.eef_position())
        self.check_contacts()
        self.observation, _, done, _ = self.env.step(command.tolist())
        self.steps += 1
        if self.recorder is not None:
            self.recorder.record(self.observation, self.steps)
        self.done = bool(done)
        self.check_workspace(self.eef_position())
        self.check_contacts()
        return self.observation

    def task_success(self) -> bool:
        return bool(self.env.check_success())

    def target_object(self, name: str) -> Any:
        objects = self.env.env.objects_dict
        if name not in objects:
            raise ValueError(f"unknown LIBERO target object: {name}")
        return objects[name]

    def grasped(self, name: str) -> bool:
        target = self.target_object(name)
        return bool(self.env.env._check_grasp(self.env.robots[0].gripper, target.contact_geoms))

    def target_in_gripper(self, name: str) -> np.ndarray:
        target = self.target_object(name)
        data = self.env.sim.data
        site = self.env.robots[0].gripper.important_sites["grip_site"]
        position = data.get_site_xpos(site)
        rotation = data.get_site_xmat(site)
        return rotation.T @ (data.get_body_xpos(target.root_body) - position)

    def capture(self, camera_utils: Any = None) -> ObservationBundle:
        if self.observation is None:
            raise RuntimeError("reset the environment before capture")
        if camera_utils is None:
            from robosuite.utils import camera_utils
        observation = self.observation
        # flip both images so depth and optical calibration share the same pixels
        rgb = np.asarray(observation[f"{self.camera}_image"])[::-1]
        raw_depth = np.asarray(observation[f"{self.camera}_depth"])[::-1]
        height, width = rgb.shape[:2]
        if rgb.shape != (height, width, 3) or width * height > 640 * 480:
            raise ValueError("invalid or over budget RGB image")
        if rgb.dtype != np.uint8 or raw_depth.size != width * height:
            raise ValueError("RGB and depth formats do not match")
        if not np.isfinite(raw_depth).all() or (raw_depth < 0).any() or (raw_depth > 1).any():
            raise ValueError("invalid normalized MuJoCo depth")
        depth = camera_utils.get_real_depth_map(self.env.sim, raw_depth).astype("<f4")
        intrinsic = camera_utils.get_camera_intrinsic_matrix(
            self.env.sim, self.camera, height, width)
        extrinsic = camera_utils.get_camera_extrinsic_matrix(self.env.sim, self.camera)
        position = self.eef_position()
        quaternion = self.eef_orientation()
        if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-4):
            raise ValueError("end effector quaternion is not normalized")
        joints = np.asarray(observation["robot0_joint_pos"], dtype=float)
        rgb_bytes, depth_bytes = rgb.tobytes(), depth.tobytes()
        descriptor = ObservationDescriptor(
            sequence=self.steps, captured_at=datetime.now(timezone.utc), world_frame="world",
            rgb_url="libero://saved/rgb", depth_url="libero://saved/depth",
            rgb_sha256=hashlib.sha256(rgb_bytes).hexdigest(),
            depth_sha256=hashlib.sha256(depth_bytes).hexdigest(), depth_encoding="32FC1",
            camera=CameraIntrinsics(width=width, height=height, fx=intrinsic[0, 0],
                                    fy=intrinsic[1, 1], cx=intrinsic[0, 2], cy=intrinsic[1, 2]),
            camera_pose=matrix_pose(extrinsic),
            end_effector_pose=Pose(position=Vector3(x=position[0], y=position[1], z=position[2]),
                orientation=Quaternion(x=quaternion[0], y=quaternion[1],
                                       z=quaternion[2], w=quaternion[3]), frame_id="world"),
            joints=JointState(names=tuple(f"robot0_joint{i + 1}" for i in range(len(joints))),
                              positions_rad=tuple(joints)),
        )
        bundle = ObservationBundle(descriptor, rgb_bytes, depth_bytes, source="libero-rgbd-only")
        bundle.validate()
        return bundle

    def start_recording(self, directory) -> None:
        from gap_libero.recording import FrameRecorder

        if self.recorder is not None or self.observation is None:
            raise RuntimeError("recording requires a ready backend and no existing recorder")
        self.recorder = FrameRecorder(directory, self.camera,
                                      max_frames=self.max_steps + 1)
        self.recorder.record(self.observation, self.steps)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                if self.recorder is not None:
                    self.recorder.close()
            finally:
                self.env.close()


def create_backend(*, suite: str = "libero_spatial", task_id: int = 0, init_state_id: int = 0,
                   seed: int = 0, **settings: Any) -> tuple[LiberoBackend, dict[str, Any]]:
    if suite not in SUITES or min(task_id, init_state_id, seed) < 0:
        raise ValueError("invalid suite or negative task state or seed")
    config_path = os.environ.get("LIBERO_CONFIG_PATH")
    if not config_path or not (Path(config_path) / "config.yaml").is_file():
        raise RuntimeError("set LIBERO_CONFIG_PATH to a prepared config directory first")
    # simulator imports remain isolated from isaac and metadata checks
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[suite]()
    if task_id >= task_suite.n_tasks:
        raise ValueError("task id is outside the selected suite")
    task = task_suite.get_task(task_id)
    states = task_suite.get_task_init_states(task_id)
    if init_state_id >= len(states):
        raise ValueError("initial state id is outside the task state list")
    camera = settings.get("camera", "agentview")
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=[camera], camera_heights=240, camera_widths=320, camera_depths=True,
        controller="OSC_POSE", control_freq=20, horizon=settings.get("max_steps", 400),
        ignore_done=False, render_gpu_device_id=0,
    )
    try:
        backend = LiberoBackend(env, **settings)
        backend.reset(seed, states[init_state_id])
    except BaseException:
        env.close()
        raise
    return backend, {"backend": "libero", "suite": suite, "task_id": task_id,
                     "task": task.name, "language": task.language, "seed": seed,
                     "init_state_id": init_state_id, "controller": "OSC_POSE"}
