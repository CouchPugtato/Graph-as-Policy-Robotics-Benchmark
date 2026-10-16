"""Native physical objects and independent checks for the custom popcorn task."""

import math

import numpy as np
from robosuite.models.objects import BoxObject, CompositeObject
from robosuite.utils.mjcf_utils import array_to_string


def make_object(item):
    if item.shape == "pan":
        obj = CompositeObject(
            name=item.object_id,
            total_size=[0.19, 0.075, 0.04],
            geom_types=["box", "box", "cylinder"],
            geom_sizes=[[0.075, 0.075, 0.018], [0.06, 0.014, 0.012], [0.075, 0.018]],
            geom_locations=[[0, 0, 0], [0.12, 0, 0.014], [0, 0, 0]],
            obj_types=["collision", "all", "visual"],
            locations_relative_to_center=True,
            geom_rgbas=[[0.3, 0.3, 0.3, 1], [*item.color_rgb, 1], [0.3, 0.3, 0.3, 1]],
            joints="default",
            density=item.mass_kg / (0.15 * 0.15 * 0.036 + 0.12 * 0.028 * 0.024),
            geom_frictions=[[1, 0.005, 0.0001]] * 3,
            solref=[0.01, 1],
        )
    else:
        joints = (
            [dict(type="hinge", axis="0 0 1", range="0 1.2", limited="true", damping=".02")]
            if item.shape == "knob"
            else None
        )
        obj = BoxObject(
            name=item.object_id,
            size=np.asarray(item.size_m) / 2,
            rgba=[*item.color_rgb, 1],
            joints=joints,
            friction=[1, 0.005, 0.0001],
        )
    obj.get_obj().set("pos", array_to_string(item.position_m))
    if item.shape == "pan":
        for geom in obj.get_obj().findall("geom"):
            if geom.get("group") == "0":
                geom.set("condim", "4")
    angle = math.radians(item.yaw_deg) / 2
    obj.get_obj().set("quat", array_to_string([math.cos(angle), 0, 0, math.sin(angle)]))
    return obj


def popcorn_evidence(env):
    data = env.sim.data
    pan, burner, knob = (env.objects_dict[key] for key in ("pan", "burner", "knob"))
    pan_position = np.asarray(data.get_body_xpos(pan.root_body))
    burner_position = np.asarray(data.get_body_xpos(burner.root_body))
    angle = float(data.get_joint_qpos(knob.joints[0]))
    held = env._check_grasp(env.robots[0].gripper, pan.contact_geoms)
    horizontal = float(np.linalg.norm(pan_position[:2] - burner_position[:2]))
    contacts = {
        frozenset((env.sim.model.geom_id2name(c.geom1), env.sim.model.geom_id2name(c.geom2)))
        for c in data.contact[: data.ncon]
    }
    supported = any(
        a in pair and b in pair
        for pair in contacts
        for a in pan.contact_geoms
        for b in burner.contact_geoms
    )
    return {
        "pan_on_burner": horizontal < 0.025 and supported,
        "pan_released": not held,
        "pan_speed_m_s": float(np.linalg.norm(data.get_body_xvelp(pan.root_body))),
        "knob_angle_rad": angle,
        "stove_on": angle >= 0.5,
        "stove_off": angle <= 0.05,
        "thermal_simulation": False,
        "heating_abstraction_steps": env.popcorn_history["heating_steps"],
        "pan_placed": env.popcorn_history["pan_placed"],
        "stove_turned_on": env.popcorn_history["stove_turned_on"],
        "stove_turned_off": env.popcorn_history["stove_turned_off"],
        "heating_wait_completed": env.popcorn_history["heating_steps"] >= 40,
    }
