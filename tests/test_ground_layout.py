from unittest.mock import Mock

import pytest

from gap_isaac.grocery_scene import GROUND_PRIM_PATH, SCENE_LAYOUT, validate_ground_scene


def fake_stage(*, floor=True, table=False, layout=SCENE_LAYOUT):
    def prim(path):
        result = Mock()
        result.IsValid.return_value = {
            GROUND_PRIM_PATH: floor,
            "/World/Task/TableTop": table,
            "/World/Evaluation": True,
        }.get(path, False)
        result.GetAttribute.return_value.Get.return_value = layout
        return result

    stage = Mock()
    stage.GetPrimAtPath.side_effect = prim
    return stage


def test_ground_layout_is_accepted():
    validate_ground_scene(fake_stage())


@pytest.mark.parametrize("settings", [{"floor": False}, {"table": True}, {"layout": None}])
def test_legacy_or_unmarked_stage_is_rejected_before_ground_resets(settings):
    with pytest.raises(RuntimeError, match="[Gg]round|ground-v1"):
        validate_ground_scene(fake_stage(**settings))
