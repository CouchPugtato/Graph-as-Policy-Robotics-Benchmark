"""Custom LIBERO task recreating the shared Isaac grocery proxy layout."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from libero.libero.envs.bddl_base_domain import register_problem
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.objects import BoxObject, CompositeObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string

from gap_isaac.config import load_task
from gap_isaac.evaluate import ObjectState, target_is_contained
from gap_isaac.grocery_scene import build_grocery_scene_plan
from gap_isaac.reset import FRANKA_HOME_POSITIONS_RAD
from gap_libero.backend import matrix_pose

ROOT = Path(__file__).resolve().parents[2]


def basket_parts(size, wall, base):
    width, depth, height = size
    return [
        ((width / 2, depth / 2, base / 2), (0, 0, base / 2)),
        ((wall / 2, depth / 2, height / 2), (-width / 2 + wall / 2, 0, base + height / 2)),
        ((wall / 2, depth / 2, height / 2), (width / 2 - wall / 2, 0, base + height / 2)),
        (((width - 2 * wall) / 2, wall / 2, height / 2),
         (0, -depth / 2 + wall / 2, base + height / 2)),
        (((width - 2 * wall) / 2, wall / 2, height / 2),
         (0, depth / 2 - wall / 2, base + height / 2)),
    ]


def camera_quaternion(eye, target):
    backward = np.asarray(eye, dtype=float) - target
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 0, 1], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, up, backward))
    q = matrix_pose(transform).orientation
    return [q.w, q.x, q.y, q.z]


@register_problem
class Graph_Grocery(SingleArmEnv):
    def __init__(self, bddl_file_name, scene_seed=0, **kwargs):
        self.task_config = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
        self.scene_plan = build_grocery_scene_plan(self.task_config, scene_seed, "alphabet_soup")
        self.objects_dict = {}
        self.obj_of_interest = ["alphabet_soup", "basket"]
        self.collisions = []
        self.initial_target_height = 0.055
        kwargs["mount_types"] = None
        super().__init__(**kwargs)

    def seed(self, seed):
        np.random.seed(seed)

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0]
        robot.robot_model.set_base_xpos(np.array([0, 0, 0]))
        robot.init_qpos = np.array(FRANKA_HOME_POSITIONS_RAD[:7])
        for actuator in robot.gripper.actuator:
            actuator.set("forcerange", "-7.2 7.2")
        arena = EmptyArena()
        arena.floor.set("type", "box")
        arena.floor.set("size", "3 3 0.05")
        arena.floor.set("pos", "0.45 0 -0.05")
        arena.floor.attrib.pop("material", None)
        arena.floor.set("rgba", "0.5 0.5 0.5 1")
        eye, target = np.array([1.35, -1.35, 1.15]), np.array([0.35, 0, 0.35])
        arena.set_camera("agentview", eye, camera_quaternion(eye, target), {"fovy": "50"})
        self.objects_dict = {}
        for item in self.scene_plan.objects:
            rgba = (*item.color_rgb, 1)
            if item.is_container:
                parts = basket_parts(item.size_m,
                    self.task_config.evaluation.basket_wall_thickness_m,
                    self.task_config.evaluation.basket_base_thickness_m)
                obj = CompositeObject(name=item.object_id, total_size=[0.12, 0.10, 0.17],
                    geom_types=["box"] * 5, geom_sizes=[part[0] for part in parts],
                    geom_locations=[part[1] for part in parts], locations_relative_to_center=True,
                    rgba=rgba, joints=None)
                obj.get_obj().set("pos", array_to_string(item.position_m))
                angle = math.radians(item.yaw_deg) / 2
                obj.get_obj().set("quat", array_to_string([math.cos(angle), 0, 0, math.sin(angle)]))
            else:
                size = ([item.size_m[0] / 2, item.size_m[2] / 2] if item.shape == "cylinder"
                        else np.asarray(item.size_m) / 2)
                constructor = CylinderObject if item.shape == "cylinder" else BoxObject
                obj = constructor(name=item.object_id, size=size, rgba=rgba,
                                  friction=[1.0, 0.005, 0.0001], solref=[0.01, 1.0])
                for geom in obj.get_obj().findall("geom"):
                    if geom.get("group") == "0":
                        geom.set("mass", str(item.mass_kg))
                        geom.set("margin", "0")
            self.objects_dict[item.object_id] = obj
        self.model = ManipulationTask(mujoco_arena=arena,
            mujoco_robots=[robot.robot_model], mujoco_objects=list(self.objects_dict.values()))

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = {name: self.sim.model.body_name2id(obj.root_body)
                            for name, obj in self.objects_dict.items()}

    def _setup_observables(self):
        observables = super()._setup_observables()
        observables["robot0_joint_pos"].set_active(True)
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        self.collisions = []
        for item in self.scene_plan.objects:
            if not item.is_container:
                angle = math.radians(item.yaw_deg) / 2
                self.sim.data.set_joint_qpos(self.objects_dict[item.object_id].joints[0],
                    [*item.position_m, math.cos(angle), 0, 0, math.sin(angle)])
        robot = self.robots[0]
        for name, position in zip(robot.gripper.joints, [0.04, -0.04], strict=True):
            self.sim.data.set_joint_qpos(name, position)
        robot.gripper.current_action = np.array([1.0, -1.0])

    def reward(self, action=None):
        return float(self._check_success())

    def containment(self):
        data = self.sim.data
        target = self.objects_dict["alphabet_soup"]
        basket = self.objects_dict["basket"]
        center = data.get_body_xpos(target.root_body)
        basket_position = data.get_body_xpos(basket.root_body)
        basket_rotation = data.get_body_xmat(basket.root_body)
        target_rotation = data.get_body_xmat(target.root_body)
        target_yaw = math.degrees(math.atan2(target_rotation[1, 0], target_rotation[0, 0]))
        basket_yaw = math.degrees(math.atan2(basket_rotation[1, 0], basket_rotation[0, 0]))
        # use the identical containment predicate rather than inventing an easier task score
        contained = target_is_contained(
            ObjectState("alphabet_soup", tuple(center), target_yaw),
            ObjectState("basket", tuple(basket_position), basket_yaw), self.task_config.evaluation)
        released = not self._check_grasp(self.robots[0].gripper, target.contact_geoms)
        speed = float(np.linalg.norm(data.get_body_xvelp(target.root_body)))
        return {"target_contained": bool(contained), "target_released": released,
                "target_speed_m_s": speed, "target_position_m": center.tolist()}

    def _check_success(self):
        values = self.containment()
        return (values["target_contained"] and values["target_released"]
                and values["target_speed_m_s"] <= 0.10)

    def forbidden_pairs(self):
        robot = self.robots[0]
        arm = list(robot.robot_model.contact_geoms)
        fingers = set(sum(robot.gripper.important_geoms.values(), []))
        hand = [name for name in robot.gripper.contact_geoms if name not in fingers]
        moving = arm + list(robot.gripper.contact_geoms)
        pairs = [(name, "floor") for name in moving if "link0" not in name]
        for name, obj in self.objects_dict.items():
            robot_geoms = arm + hand if name == "alphabet_soup" else moving
            pairs.extend((first, second) for first in robot_geoms for second in obj.contact_geoms)
        return pairs
