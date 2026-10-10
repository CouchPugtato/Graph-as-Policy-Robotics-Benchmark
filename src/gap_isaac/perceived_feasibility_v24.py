"""Opt-in XY surface-center adapter; v23 planning and safety checks stay frozen."""
from dataclasses import replace

from gap_isaac import perceived_feasibility as baseline
from gap_isaac.cylinder_center import verified_center


def prepare_checks(target, basket, wall_m, margin_m):
    checks = baseline.prepare_checks(target, basket, wall_m, margin_m)
    xy = verified_center(target)
    for name in ("align", "grasp", "lift"):
        checks["waypoints_m"][name][:2] = xy
    checks["cylinder_center"] = target["cylinder_center"]
    return checks


def build_perceived_grasp_plan(target, basket, wall_m, margin_m):
    plan, _ = baseline.build_perceived_grasp_plan(target, basket, wall_m, margin_m)
    checks = prepare_checks(target, basket, wall_m, margin_m)
    xy = checks["waypoints_m"]["grasp"][:2]
    updates = {name: replace(getattr(plan, name), x_m=xy[0], y_m=xy[1])
               for name in ("target_center", "align", "grasp", "lift")}
    return replace(plan, **updates), checks


validate_grasp_gate = baseline.validate_grasp_gate
