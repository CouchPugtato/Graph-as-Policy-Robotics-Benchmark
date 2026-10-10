"""Run with the existing NumPy-bearing perception interpreter; no models or Isaac."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gap_isaac.cylinder_center import augment, fit_center, verified_center
from gap_isaac.perceived_feasibility import prepare_checks as baseline_checks
from gap_isaac.perceived_feasibility_v24 import prepare_checks


def cloud(center=(.60, .14), arc=np.pi):
    angles = np.linspace(-arc / 2, arc / 2, 120)
    rng = np.random.default_rng(5)
    xy = np.array(center) + .035 * np.c_[np.cos(angles), np.sin(angles)]
    xy += rng.normal(0, .0001, xy.shape)
    side = np.c_[xy, rng.uniform(.020, .098, len(xy))]
    top = np.c_[np.array(center) + rng.uniform(-.02, .02, (40, 2)), np.full(40, .110)]
    floor = np.c_[np.array(center) + rng.uniform(-.02, .02, (20, 2)), np.zeros(20)]
    return np.r_[side, top, floor]


def geometry(object_id, dimensions, center):
    return {"ok": True, "motion_enabled": False, "target_id": object_id,
            "bundle": "capture", "image_sha256": "i", "depth_sha256": "d",
            "sequence": 1, "world_frame": "world", "units": "meters", "skills_revision": "s",
            "point_count": 180, "valid_mask_depth_fraction": 1,
            "reference_deprojection_max_error_m": 0, "obb_extent_convention": "half-extents",
            "visible_surface_dimensions_m": dimensions,
            "obb": {"center": dict(zip("xyz", center, strict=True)),
                    "extent": dict(zip("xyz", [d / 2 for d in dimensions], strict=True)),
                    "orientation": {"w": 1, "x": 0, "y": 0, "z": 0}}}


class CylinderCases(unittest.TestCase):
    def test_recovers_partial_sidewall_center_with_surface_contamination(self):
        points = cloud()
        points = np.r_[points, [[.65, .18, .05], [.63, .17, .04]]]
        fit = fit_center(points, [.604, .135])
        np.testing.assert_allclose(fit["center_xy_m"], [.60, .14], atol=.0002)
        self.assertFalse(fit["uses_evaluator_pose"])

    def test_translation_and_rotation_have_no_directional_offset(self):
        points = cloud()
        theta = .9
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        translation = np.array([-.23, .30])
        moved = points.copy()
        moved[:, :2] = points[:, :2] @ rotation.T + translation
        initial = np.array([.604, .135]) @ rotation.T + translation
        fit = fit_center(moved, initial)
        np.testing.assert_allclose(fit["center_xy_m"],
                                   np.array([.60, .14]) @ rotation.T + translation, atol=.0002)

    def test_insufficient_arc_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_center(cloud(arc=.3), [.601, .139])

    def test_nonfinite_and_sparse_clouds_are_rejected(self):
        for points in (np.zeros((9, 3)), np.full((100, 3), np.nan)):
            with self.assertRaises(ValueError):
                fit_center(points, [.60, .14])

    def test_plane_and_incorrect_radius_are_rejected(self):
        rng = np.random.default_rng(3)
        plane = np.c_[rng.uniform(.575, .625, 120), np.full(120, .14),
                      rng.uniform(.02, .1, 120)]
        bad_radius = cloud()
        bad_radius[:, :2] = (bad_radius[:, :2] - [.6, .14]) * 1.8 + [.6, .14]
        for points in (plane, bad_radius):
            with self.assertRaises(ValueError):
                fit_center(points, [.60, .14])

    def test_raw_gate_and_z_waypoints_remain_unchanged(self):
        target = geometry("alphabet_soup", [.065, .060, .11], [.604, .135, .058])
        basket = geometry("basket", [.24, .20, .16], [.4, .3, .085])
        original = baseline_checks(target, basket, .015, .005)
        with patch("gap_isaac.perceived_feasibility_v24.verified_center", return_value=(.6, .14)):
            target["cylinder_center"] = {"test": True}
            new = prepare_checks(target, basket, .015, .005)
        for name in original["waypoints_m"]:
            self.assertEqual(original["waypoints_m"][name][2], new["waypoints_m"][name][2])
            if name in ("align", "grasp", "lift"):
                self.assertEqual(new["waypoints_m"][name][:2], [.6, .14])
            else:
                self.assertEqual(original["waypoints_m"][name], new["waypoints_m"][name])
        for key in ("finger_aperture_fit", "opening_clearance_after_margin_m", "pair_checks"):
            self.assertEqual(original[key], new[key])

    def test_bad_raw_dimensions_cannot_be_hidden_by_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = geometry("alphabet_soup", [.155, .062, .17], [.60, .14, .085])
            basket = geometry("basket", [.24, .20, .16], [.4, .3, .085])
            (root / "target.json").write_text(json.dumps(target))
            (root / "basket.json").write_text(json.dumps(basket))
            np.save(root / "world_points.npy", cloud())
            with self.assertRaisesRegex(ValueError, "more than 30%"):
                augment(root / "target.json", root / "basket.json", root / "candidate.json")
            self.assertFalse((root / "candidate.json").exists())

    def test_receipt_recomputation_rejects_edited_center(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = geometry("alphabet_soup", [.065, .060, .11], [.604, .135, .058])
            basket = geometry("basket", [.24, .20, .16], [.4, .3, .085])
            (root / "target.json").write_text(json.dumps(target))
            (root / "basket.json").write_text(json.dumps(basket))
            np.save(root / "world_points.npy", cloud())
            augment(root / "target.json", root / "basket.json", root / "candidate.json")
            receipt = json.loads((root / "candidate.json").read_text())
            verified_center(receipt)
            edited = copy.deepcopy(receipt)
            edited["cylinder_center"]["center_xy_m"][1] += .003
            with self.assertRaisesRegex(ValueError, "differs"):
                verified_center(edited)


if __name__ == "__main__":
    unittest.main(verbosity=2)
