"""bounded geometric pan registration and robot pose planning"""

from pathlib import Path

import numpy as np


def configure_pan_gripper(env):
    # keep robot contact changes separate from the published object assets
    model = env.sim.model
    actuators = env.robots[0]._ref_joint_gripper_actuator_indexes
    model.actuator_gainprm[actuators, 0] *= 2
    model.actuator_biasprm[actuators, 1] *= 2
    for name in model.geom_names:
        if name and name.startswith("gripper0_") and name.endswith("pad_collision"):
            model.geom_condim[model.geom_name2id(name)] = 4


def load_pan_template(mesh_path: Path):
    raw = mesh_path.read_bytes()
    counts = np.frombuffer(raw, dtype="<i4", count=4)
    if not 100 <= counts[0] <= 100000:
        raise ValueError("pan template exceeds its vertex budget")
    vertices = np.frombuffer(raw, dtype="<f4", count=int(counts[0]) * 3, offset=16).reshape(-1, 3)
    return vertices


def fit_pan_pose(points, vertices, center, heading, floor, *, handle_extent=0.28):
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    tree = cKDTree(vertices)
    source = np.asarray(points, dtype=float)
    delta = source[:, :2] - center
    axis = np.array([np.cos(heading), np.sin(heading)])
    along = delta @ axis
    across = np.abs(delta @ np.array([-axis[1], axis[0]]))
    selected = np.linalg.norm(delta, axis=1) < 0.115
    selected |= (along > 0.1) & (along < handle_extent) & (across < 0.035)
    cloud = source[selected]
    cloud = cloud[:: max(1, len(cloud) // 1800)]
    initial = np.array([*center, floor + 0.003, 0, 0, heading])

    def distances(pose, observed):
        rotation = Rotation.from_euler("xyz", pose[3:]).as_matrix()
        local = (observed - pose[:3]) @ rotation
        return tree.query(local, workers=1)[0]

    cloud = cloud[distances(initial, cloud) < 0.035]
    if len(cloud) < 300:
        raise ValueError("insufficient pan template correspondence")
    bounds = np.array([0.04, 0.04, 0.04, 0.45, 0.45, 0.3])
    fit = least_squares(
        distances,
        initial,
        args=(cloud,),
        bounds=(initial - bounds, initial + bounds),
        loss="soft_l1",
        f_scale=0.002,
        max_nfev=100,
        x_scale=[0.02, 0.02, 0.02, 0.2, 0.2, 0.2],
    )
    residual = distances(fit.x, cloud)
    inliers = residual < 0.005
    if inliers.mean() < 0.9 and handle_extent > 0.19:
        return fit_pan_pose(points, vertices, center, heading, floor, handle_extent=0.19)
    if np.median(residual) > 0.004 or inliers.mean() < 0.65:
        raise ValueError("pan template registration was not verified")
    rotation = Rotation.from_euler("xyz", fit.x[3:]).as_matrix()
    return {
        "body_position_m": fit.x[:3].tolist(),
        "body_rotation": rotation.tolist(),
        "fit_median_error_m": float(np.median(residual)),
        "fit_inlier_fraction": float(inliers.mean()),
        "fit_points": len(cloud),
        "registration_handle_extent_m": handle_extent,
    }


class PanPosePlanner:
    def __init__(self, backend, floor, stove_center, stove_top):
        import mujoco
        from scipy.spatial import ConvexHull

        self.mujoco = mujoco
        self.backend = backend
        self.model = backend.env.sim.model._model
        # plan on a separate state using observed poses and robot proprioception
        self.data = mujoco.MjData(self.model)
        self.indices = backend.env.robots[0]._ref_joint_pos_indexes
        self.joints = backend.env.robots[0]._ref_joint_indexes
        self.current = backend.env.sim.data.qpos[self.indices].copy()
        self.data.qpos[self.indices] = self.current
        fingers = backend.env.robots[0]._ref_gripper_joint_pos_indexes
        self.data.qpos[fingers] = [0.022, -0.022]
        self.site = backend.env.sim.model.site_name2id(
            backend.env.robots[0].gripper.important_sites["grip_site"]
        )
        self.floor = floor
        self.rejections = {"unreachable": 0, "collision": 0, "path": 0}
        self.boxes = [
            (
                np.array([*stove_center, (floor + stove_top) / 2]),
                np.array([0.105, 0.105, (stove_top - floor) / 2 + 0.007]),
                np.eye(3),
                False,
            )
        ]
        self.shapes = []
        self.gripper_geoms = set()
        for name in backend.env.sim.model.geom_names:
            if not name:
                continue
            i = backend.env.sim.model.geom_name2id(name)
            if self.model.geom_group[i] != 0:
                continue
            if not (name.startswith("gripper0_") or name.startswith("robot0_")):
                continue
            if "link0" in name or "link1" in name:
                continue
            if name.startswith("gripper0_"):
                self.gripper_geoms.add(i)
            if self.model.geom_type[i] == 7:
                mesh = self.model.geom_dataid[i]
                start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
                vertices = self.model.mesh_vert[start : start + count]
            elif self.model.geom_type[i] == 6:
                vertices = (
                    np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
                    * self.model.geom_size[i]
                )
            else:
                continue
            hull = ConvexHull(vertices)
            edge_pairs = np.vstack(
                [hull.simplices[:, [0, 1]], hull.simplices[:, [1, 2]], hull.simplices[:, [0, 2]]]
            )
            edges = vertices[edge_pairs[:, 1]] - vertices[edge_pairs[:, 0]]
            edges /= np.maximum(np.linalg.norm(edges, axis=1, keepdims=True), 1e-9)
            edges = np.unique(np.round(edges, 3), axis=0)
            self.shapes.append((i, vertices[hull.vertices], hull.equations[:, :3], edges))

    def forward(self, q):
        self.data.qpos[self.indices] = q
        self.mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(
            3, 3
        ).copy()

    def clear(self, q):
        self.forward(q)
        for index, points, normals, edges in self.shapes:
            rotation = self.data.geom_xmat[index].reshape(3, 3)
            world = points @ rotation.T + self.data.geom_xpos[index]
            if index in self.gripper_geoms and world[:, 2].min() < self.floor + 0.001:
                return False
            for center, half, box_rotation, target in self.boxes:
                if target and index not in self.gripper_geoms:
                    continue
                box_world_half = np.abs(box_rotation) @ half
                if np.any(world.max(0) < center - box_world_half) or np.any(
                    world.min(0) > center + box_world_half
                ):
                    continue
                directions = edges @ rotation.T @ box_rotation
                axes = np.vstack(
                    [
                        np.eye(3),
                        normals @ rotation.T @ box_rotation,
                        np.cross(directions[:, None, :], np.eye(3)[None, :, :]).reshape(-1, 3),
                    ]
                )
                axes = axes[np.linalg.norm(axes, axis=1) > 1e-6]
                projected = ((world - center) @ box_rotation) @ axes.T
                radius = np.abs(axes) @ half
                separated = (projected.min(0) > radius + 0.001) | (
                    projected.max(0) < -radius - 0.001
                )
                if not separated.any():
                    return False
        return True

    def solve(self, position, rotation, seed):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        low, high = self.model.jnt_range[self.joints].T

        def residual(q):
            xyz, matrix = self.forward(q)
            angle = Rotation.from_matrix(rotation @ matrix.T).as_rotvec()
            return np.r_[xyz - position, 0.2 * angle, 0.0001 * (q - seed)]

        result = least_squares(
            residual,
            np.clip(seed, low + 0.01, high - 0.01),
            bounds=(low + 0.01, high - 0.01),
            max_nfev=100,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )
        error = residual(result.x)
        if np.linalg.norm(error[:3]) > 0.002 or np.linalg.norm(error[3:6]) > 0.006:
            self.rejections["unreachable"] += 1
            return None
        if not self.clear(result.x):
            self.rejections["collision"] += 1
            return None
        return result.x

    def choose(self, fit):
        from scipy.spatial.transform import Rotation

        body = np.asarray(fit["body_position_m"])
        matrix = np.asarray(fit["body_rotation"])
        model = self.backend.env.sim.model
        for name in self.backend.target_object("chefmate_8_frypan").contact_geoms:
            index = model.geom_name2id(name)
            if model.geom_type[index] != 6:
                continue
            quaternion = model.geom_quat[index]
            local_rotation = Rotation.from_quat([*quaternion[1:], quaternion[0]]).as_matrix()
            self.boxes.append(
                (
                    body + matrix @ model.geom_pos[index],
                    model.geom_size[index] + 0.002,
                    matrix @ local_rotation,
                    True,
                )
            )
        candidates = []
        base = self.backend.env.sim.data.get_body_xpos(
            self.backend.env.robots[0].robot_model.root_body
        )
        outward = np.r_[body[:2] - base[:2], 0]
        outward /= np.linalg.norm(outward)
        tilt_axis = np.cross([0, 0, -1], outward)
        import time
        from itertools import product

        deadline = time.monotonic() + 60
        for height, angle, tilt in product(
            (0, 0.004, -0.006, 0.01),
            (
                0,
                np.pi / 36,
                -np.pi / 36,
                np.pi / 18,
                -np.pi / 18,
                np.pi / 12,
                -np.pi / 12,
                np.pi / 6,
                -np.pi / 6,
                np.pi / 4,
                -np.pi / 4,
            ),
            (0, np.pi / 18, np.pi / 15, np.pi / 12, np.pi / 6, np.pi / 4),
        ):
            for radius, flip, azimuth in product(
                (0.15, 0.17, 0.19, 0.21),
                (0, np.pi),
                (0,) if tilt == 0 else (0, np.pi / 2, np.pi, -np.pi / 2),
            ):
                if time.monotonic() > deadline:
                    raise ValueError(f"pan pose search exceeded its time budget: {self.rejections}")
                local = (
                    Rotation.from_euler("z", np.pi / 2 + angle + flip).as_matrix()
                    @ Rotation.from_euler("x", np.pi).as_matrix()
                )
                axis = Rotation.from_euler("z", azimuth).apply(tilt_axis)
                orientation = Rotation.from_rotvec(tilt * axis).as_matrix() @ matrix @ local
                contact = body + matrix @ [radius, 0, 0.044 + (radius - 0.1926) * 0.128]
                position = contact - orientation @ [0, 0, -0.0036]
                position[2] += height
                q = self.solve(position, orientation, self.current)
                if q is None:
                    continue
                hover = position.copy()
                hover[2] = max(position[2] + 0.065, self.floor + 0.13)
                above = self.solve(hover, orientation, q)
                if above is None:
                    continue
                if not all(self.clear(above * (1 - t) + q * t) for t in np.linspace(0, 1, 15)):
                    self.rejections["path"] += 1
                    continue
                cost = (
                    abs(angle),
                    tilt,
                    abs(height),
                    np.linalg.norm(above - self.current) + abs(radius - 0.15),
                )
                candidates.append((cost, position, hover, orientation, radius, q, above))
            if candidates:
                break
        if not candidates:
            raise ValueError(f"no reachable collision clear pan grasp was found: {self.rejections}")
        cost, position, hover, rotation, radius, q, above = min(candidates, key=lambda c: c[0])
        return {
            "position_m": position.tolist(),
            "hover_m": hover.tolist(),
            "orientation_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
            "model_grasp_offset_m": [radius, 0, 0.044 + (radius - 0.1926) * 0.128],
            "joint_grasp": q.tolist(),
            "joint_hover": above.tolist(),
            "candidates_verified": len(candidates),
            "jaw_skew_rad": cost[0],
            "wrist_tilt_rad": cost[1],
        }


def settle_pan_grasp(bindings):
    from collections import deque

    from scipy.spatial.transform import Rotation

    from gap_isaac.graph_policy import NodeResult

    backend = bindings.backend
    robot = backend.env.robots[0]
    data = backend.env.sim.data
    actuators = robot._ref_joint_gripper_actuator_indexes
    history = deque(maxlen=10)
    bindings.gripper = 1.0
    for _ in range(bindings.plan.actions["close_gripper"].max_steps):
        bindings.command()
        if (
            np.max(np.abs(data.ctrl[actuators])) > 0.001
            or np.min(np.abs(data.actuator_force[actuators])) < 5
        ):
            history.clear()
            continue
        if not backend.grasped(bindings.plan.target_object):
            history.clear()
            continue
        point = bindings.grasp_point()
        rotation = backend.target_rotation_in_gripper(bindings.plan.target_object)
        history.append((point.copy(), rotation.copy()))
        if len(history) < history.maxlen:
            continue
        movement = max(np.linalg.norm(p - point) for p, r in history)
        turning = max(
            np.linalg.norm(Rotation.from_matrix(r @ rotation.T).as_rotvec()) for p, r in history
        )
        if movement > 0.001 or turning > 0.015:
            continue
        # establish the hold reference only after the clamp has settled
        bindings.hold_reference = point.copy()
        bindings.hold_rotation = rotation.copy()
        bindings.missing_contact_frames = 0
        for _ in range(3):
            bindings.command(require_grasp=True)
        return NodeResult(detail="two pad grasp verified after physical settling")
    return NodeResult("failed", "pan grasp did not stabilize before its step budget")


def preshape_pan_gripper(bindings):
    backend = bindings.backend
    robot = backend.env.robots[0]
    data = backend.env.sim.data
    for _ in range(160):
        targets = data.ctrl[robot._ref_joint_gripper_actuator_indexes]
        bindings.gripper = 1.0 if np.max(np.abs(targets)) > 0.022 else 0.0
        bindings.command()
        positions = data.qpos[robot._ref_gripper_joint_pos_indexes]
        if bindings.gripper == 0 and np.max(np.abs(positions)) <= 0.023:
            return
    raise RuntimeError("pan gripper preshape did not settle within its step budget")
