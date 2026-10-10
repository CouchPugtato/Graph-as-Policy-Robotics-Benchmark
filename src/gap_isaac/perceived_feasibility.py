"""Conservative, model-assisted checks and proposed poses; never executes motion."""
from __future__ import annotations

import math


def grasp_slip_check_active(physical_test: bool, perceived: bool, holding_phase: bool,
                            has_reference: bool) -> bool:
    return (physical_test or perceived) and holding_phase and has_reference


def prepare_checks(target: dict, basket: dict, wall_m: float, margin_m: float) -> dict:
    from gap_isaac.geometry_pair_check import check_pair
    pair = check_pair(target, basket)
    if not (0 < wall_m < .05 and 0 <= margin_m < .05):
        raise ValueError("Invalid basket wall/clearance prior")
    target_dims = target["visible_surface_dimensions_m"]
    basket_dims = basket["visible_surface_dimensions_m"]
    opening = [size - 2 * wall_m for size in basket_dims[:2]]
    # Conservative cylinder footprint compensates for hidden single-view faces.
    # The 70 mm prior belongs to this synthetic target, not a learned estimate.
    diameter = max(.070, *target_dims[:2]) + .010
    required = diameter + 2 * margin_m
    clearance = [size - required for size in opening]
    opening_ok = all(math.isfinite(value) and value > 0 for value in clearance)
    gripper_max_aperture = .080
    finger_fit = max(.070, *target_dims[:2]) + .004 < gripper_max_aperture
    tc = target["obb"]["center"]
    bc = basket["obb"]["center"]
    height = target_dims[2]
    # Preserve the physical-smoke TCP convention: 100 mm forward of link8,
    # with the near object face 70 mm forward of the palm.
    grasp_offset = .070 + height / 2 - .100
    grasp_z = tc["z"] + grasp_offset
    rim_z = bc["z"] + basket["obb"]["extent"]["z"]
    release_z = rim_z + .04 + height / 2 + grasp_offset
    transit_z = max(.30, rim_z + .13, release_z)
    waypoints = {"align": [tc["x"], tc["y"], transit_z],
                 "grasp": [tc["x"], tc["y"], grasp_z],
                 "lift": [tc["x"], tc["y"], transit_z],
                 "transport": [bc["x"], bc["y"], transit_z],
                 "release_above_rim": [bc["x"], bc["y"], release_z]}
    return {"pair_checks": pair, "opening_fit_model_assisted": opening_ok,
            "opening_dimensions_estimate_m": opening, "conservative_target_footprint_m": diameter,
            "opening_clearance_after_margin_m": clearance, "wall_thickness_prior_m": wall_m,
            "containment_margin_m": margin_m, "finger_aperture_fit": finger_fit,
            "gripper_max_aperture_m": gripper_max_aperture, "waypoints_m": waypoints,
            "grasp_offset_m": grasp_offset, "rim_z_estimate_m": rim_z,
            "orientation_wxyz": [0, 0, 1, 0], "graph_motion_ready": False,
            "limitations": ["Opening estimate uses known wall thickness, not an independently perceived interior.",
                            "Waypoints use a single-view fitted object center; contact stability remains untested.",
                            "IK feasibility does not imply a collision-free path."]}


def build_perceived_grasp_plan(target: dict, basket: dict, wall_m: float, margin_m: float):
    from gap_isaac.pick_place_plan import CartesianWaypoint, OraclePickPlacePlan
    checks = prepare_checks(target, basket, wall_m, margin_m)
    if not checks["opening_fit_model_assisted"] or not checks["finger_aperture_fit"]:
        raise ValueError("Perceived geometry fails opening/aperture checks")
    waypoints = checks["waypoints_m"]
    center = target["obb"]["center"]
    # Reuse the waypoint data container, not the oracle builder or reset poses.
    plan = OraclePickPlacePlan(
        target_center=CartesianWaypoint(*(center[axis] for axis in ("x", "y", "z"))),
        grasp=CartesianWaypoint(*waypoints["grasp"]), align=CartesianWaypoint(*waypoints["align"]),
        lift=CartesianWaypoint(*waypoints["lift"]), transport=CartesianWaypoint(*waypoints["transport"]),
        place=CartesianWaypoint(*waypoints["release_above_rim"]), retract=CartesianWaypoint(*waypoints["transport"]),
        grasp_frame_above_object_center_m=checks["grasp_offset_m"],
        basket_rim_z_m=checks["rim_z_estimate_m"], release_bottom_clearance_m=.04)
    return plan, checks


def validate_grasp_gate(result: dict, waypoints: dict) -> None:
    if not result.get("ok") or result.get("test") != "physical-grasp":
        raise ValueError("A successful physical grasp trial is required")
    if result.get("geometry_source") != "saved-dino-sam3-rgbd-model-assisted":
        raise ValueError("Oracle grasp trials cannot authorize perceived graph execution")
    diagnostics = result.get("diagnostics", {})
    perceived = diagnostics.get("perceived_geometry", {})
    if perceived.get("pose_corrections_from_oracle") is not False:
        raise ValueError("Grasp gate must not use oracle pose correction")
    if perceived.get("checks", {}).get("waypoints_m") != waypoints:
        raise ValueError("Grasp gate geometry differs from the proposed graph waypoints")
    slip = result.get("maximum_relative_displacement_m", float("inf"))
    if not math.isfinite(slip) or not 0 <= slip <= .015:
        raise ValueError("Grasp gate exceeds the unchanged slip bound")
    phases = {item["phase"]: item for item in diagnostics.get("physical_grasp_checkpoints", [])}
    for phase in ("close", "lift", "stationary_hold", "lateral_move", "final_hold"):
        if phases.get(phase, {}).get("held_object_id") != "alphabet_soup":
            raise ValueError("Grasp gate lacks verified hold checkpoints")
