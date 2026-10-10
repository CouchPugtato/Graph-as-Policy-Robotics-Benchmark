from copy import deepcopy

import pytest

from gap_isaac.geometry_pair_check import check_pair
from gap_isaac.grocery_scene import GEOMETRY


def make_geometry(object_id):
    dimensions = list(GEOMETRY[object_id].size_m)
    return {"target_id": object_id, "bundle": "same-bundle", "image_sha256": "same-rgb",
            "depth_sha256": "same-depth", "sequence": 127, "world_frame": "world",
            "units": "meters", "skills_revision": "same-pin", "ok": True,
            "motion_enabled": False, "point_count": 206, "valid_mask_depth_fraction": 1.0,
            "reference_deprojection_max_error_m": 1e-8,
            "visible_surface_dimensions_m": dimensions, "obb_extent_convention": "half-extents",
            "obb": {"center": {"x": .7 if object_id == "alphabet_soup" else .3,
                               "y": .15 if object_id == "alphabet_soup" else -.2,
                               "z": dimensions[2] / 2},
                    "extent": dict(zip(("x", "y", "z"), (value / 2 for value in dimensions))),
                    "orientation": {"w": 1, "x": 0, "y": 0, "z": 0}}}


def test_valid_pair_never_authorizes_motion():
    result = check_pair(make_geometry("alphabet_soup"), make_geometry("basket"))
    assert result["ok"]
    assert not result["graph_motion_ready"]
    assert len(result["remaining_checks"]) == 3


@pytest.mark.parametrize("key,value,match", [
    ("sequence", 128, "mismatch"),
    ("units", "millimeters", "mismatch"),
    ("visible_surface_dimensions_m", [1, 1, 1], "proxy size"),
])
def test_inconsistent_pair_is_rejected(key, value, match):
    basket = deepcopy(make_geometry("basket"))
    basket[key] = value
    with pytest.raises(ValueError, match=match):
        check_pair(make_geometry("alphabet_soup"), basket)
