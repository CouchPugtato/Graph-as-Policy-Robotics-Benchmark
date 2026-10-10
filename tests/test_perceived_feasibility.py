import pytest

from gap_isaac.perceived_feasibility import build_perceived_grasp_plan, prepare_checks, validate_grasp_gate, grasp_slip_check_active


def geometry(dimensions, center):
    return {"visible_surface_dimensions_m": dimensions,
            "obb": {"center": dict(zip(("x", "y", "z"), center)),
                    "extent": dict(zip(("x", "y", "z"), (v / 2 for v in dimensions)))}}


def test_opening_estimate_uses_walls_and_margin_and_never_authorizes_motion(monkeypatch):
    monkeypatch.setattr("gap_isaac.geometry_pair_check.check_pair", lambda *args: {"ok": True})
    result = prepare_checks(geometry([.064, .062, .107], [.69, .15, .059]),
                            geometry([.24, .20, .163], [.37, -.25, .091]), .015, .005)
    assert result["opening_dimensions_estimate_m"] == pytest.approx([.21, .17])
    assert result["opening_clearance_after_margin_m"] == pytest.approx([.12, .08])
    assert result["opening_fit_model_assisted"]
    assert result["finger_aperture_fit"]
    assert not result["graph_motion_ready"]
    assert result["waypoints_m"]["grasp"][2] == pytest.approx(.0825)


def test_too_narrow_opening_fails_without_increasing_tolerance(monkeypatch):
    monkeypatch.setattr("gap_isaac.geometry_pair_check.check_pair", lambda *args: {"ok": True})
    result = prepare_checks(geometry([.07, .07, .11], [.69, .15, .055]),
                            geometry([.1, .1, .15], [.37, -.25, .075]), .015, .005)
    assert not result["opening_fit_model_assisted"]


def test_oversized_object_does_not_fit_fingers(monkeypatch):
    monkeypatch.setattr("gap_isaac.geometry_pair_check.check_pair", lambda *args: {"ok": True})
    result = prepare_checks(geometry([.08, .08, .11], [.69, .15, .055]),
                            geometry([.24, .20, .15], [.37, -.25, .075]), .015, .005)
    assert not result["finger_aperture_fit"]


def test_perceived_plan_uses_fitted_centers_not_reset_poses(monkeypatch):
    monkeypatch.setattr("gap_isaac.geometry_pair_check.check_pair", lambda *args: {"ok": True})
    target = geometry([.064, .062, .107], [.69, .15, .059])
    basket = geometry([.24, .20, .163], [.37, -.25, .091])
    plan, checks = build_perceived_grasp_plan(target, basket, .015, .005)
    assert plan.target_center.as_tuple() == (.69, .15, .059)
    assert plan.grasp.as_tuple() == pytest.approx([.69, .15, .0825])
    assert plan.transport.x_m == .37
    assert plan.transport.y_m == -.25
    assert not checks["graph_motion_ready"]


def test_perceived_plan_refuses_failed_opening_check(monkeypatch):
    monkeypatch.setattr("gap_isaac.geometry_pair_check.check_pair", lambda *args: {"ok": True})
    with pytest.raises(ValueError, match="opening/aperture"):
        build_perceived_grasp_plan(geometry([.07, .07, .11], [.69, .15, .055]),
                                   geometry([.1, .1, .15], [.37, -.25, .075]), .015, .005)


def successful_gate():
    return {"ok": True, "test": "physical-grasp", "maximum_relative_displacement_m": .001,
            "geometry_source": "saved-dino-sam3-rgbd-model-assisted",
            "diagnostics": {"perceived_geometry": {"pose_corrections_from_oracle": False,
                                                    "checks": {"waypoints_m": {"grasp": [1, 2, 3]}}},
                            "physical_grasp_checkpoints": [{"phase": phase, "held_object_id": "alphabet_soup"}
                                for phase in ("close", "lift", "stationary_hold", "lateral_move", "final_hold")]}}


def test_grasp_gate_checks_provenance_waypoints_and_hold():
    validate_grasp_gate(successful_gate(), {"grasp": [1, 2, 3]})
    with pytest.raises(ValueError, match="differs"):
        validate_grasp_gate(successful_gate(), {"grasp": [4, 5, 6]})


@pytest.mark.parametrize("key,value", [("ok", False), ("geometry_source", "oracle-smoke-only"),
                                      ("maximum_relative_displacement_m", .016)])
def test_failed_or_oracle_grasp_gate_cannot_enable_graph(key, value):
    result = successful_gate()
    result[key] = value
    with pytest.raises(ValueError):
        validate_grasp_gate(result, {"grasp": [1, 2, 3]})


def test_slip_monitor_is_active_during_hold_but_not_after_intentional_release():
    assert grasp_slip_check_active(False, True, True, True)
    assert grasp_slip_check_active(True, False, True, True)
    assert not grasp_slip_check_active(False, True, False, True)
    assert not grasp_slip_check_active(False, True, True, False)
