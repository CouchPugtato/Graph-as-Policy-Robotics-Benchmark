"""Deterministic Cartesian waypoints for the first oracle-geometry graph smoke."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from gap_isaac.config import TaskConfig
from gap_isaac.grocery_scene import GEOMETRY, WORK_SURFACE_Z_M
from gap_isaac.reset import EpisodeResetPlan

RELEASE_BOTTOM_CLEARANCE_M = 0.04


@dataclass(frozen=True)
class CartesianWaypoint:
    x_m: float
    y_m: float
    z_m: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x_m, self.y_m, self.z_m)


@dataclass(frozen=True)
class OraclePickPlacePlan:
    target_center: CartesianWaypoint
    grasp: CartesianWaypoint
    align: CartesianWaypoint
    lift: CartesianWaypoint
    transport: CartesianWaypoint
    place: CartesianWaypoint
    retract: CartesianWaypoint
    grasp_frame_above_object_center_m: float
    basket_rim_z_m: float
    release_bottom_clearance_m: float


def build_oracle_pick_place_plan(
    task: TaskConfig,
    reset: EpisodeResetPlan,
    settled_positions: Mapping[str, tuple[float, float, float]] | None = None,
) -> OraclePickPlacePlan:
    """Build a conservative top-down path from reset poses for simulator smoke testing.

    This deliberately uses reset geometry and must not be presented as policy perception.
    It is replaced by open-skill perception outputs in the evaluated graph.
    """

    objects = {item.object_id: item for item in reset.objects}
    target = objects[reset.target_id]
    container = objects[reset.container_id]
    live_positions = dict(settled_positions or {})
    target_position = live_positions.get(reset.target_id, target.position_m)
    container_position = live_positions.get(reset.container_id, container.position_m)
    target_geometry = GEOMETRY[reset.target_id]
    container_geometry = GEOMETRY[reset.container_id]

    # Lula's Franka ``right_gripper`` frame is 100 mm forward of panda_link8.
    # The collision-safe finger region begins with the object's near face
    # 70 mm forward of the hand, beyond the palm collision mesh (~66 mm).
    # This provides 5 mm more finger overlap than the initial contact-only grasp.
    # Preserve that grasp-frame-to-object transform through release instead
    # of driving the palm to the object centre or into the basket walls.
    hand_to_grasp_frame_m = 0.100
    palm_to_object_near_face_m = 0.070
    grasp_frame_above_object_center_m = (
        palm_to_object_near_face_m + target_geometry.size_m[2] / 2 - hand_to_grasp_frame_m
    )

    # Walls start above the base slab, so their top includes base thickness.
    basket_rim_z = (
        container_position[2]
        + task.evaluation.basket_base_thickness_m
        + container_geometry.size_m[2]
    )
    place_z = (
        basket_rim_z
        + RELEASE_BOTTOM_CLEARANCE_M
        + target_geometry.size_m[2] / 2
        + grasp_frame_above_object_center_m
    )
    clearance_z = max(
        WORK_SURFACE_Z_M + 0.30,
        target_position[2] + 0.20,
        basket_rim_z + 0.13,
        place_z,
    )
    target_center = CartesianWaypoint(*target_position)
    grasp = CartesianWaypoint(
        target_position[0],
        target_position[1],
        target_position[2] + grasp_frame_above_object_center_m,
    )
    return OraclePickPlacePlan(
        target_center=target_center,
        grasp=grasp,
        align=CartesianWaypoint(target_position[0], target_position[1], clearance_z),
        lift=CartesianWaypoint(target_position[0], target_position[1], clearance_z),
        transport=CartesianWaypoint(container_position[0], container_position[1], clearance_z),
        place=CartesianWaypoint(
            container_position[0],
            container_position[1],
            place_z,
        ),
        retract=CartesianWaypoint(container_position[0], container_position[1], clearance_z),
        grasp_frame_above_object_center_m=grasp_frame_above_object_center_m,
        basket_rim_z_m=basket_rim_z,
        release_bottom_clearance_m=RELEASE_BOTTOM_CLEARANCE_M,
    )
