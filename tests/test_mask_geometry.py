import pytest

from gap_isaac.mask_geometry import validate_cloud_summary


def test_small_ground_target_cloud_is_accepted():
    validate_cloud_summary(206, [.65, .10, .001], [.72, .17, .11])


@pytest.mark.parametrize("count,minimum,maximum,match", [
    (9, [0, 0, 0], [.1, .1, .1], "ten"),
    (20, [0, 0, float("nan")], [.1, .1, .1], "nonfinite"),
    (20, [0, 0, -.03], [.1, .1, .1], "below"),
    (20, [0, 0, 0], [.6, .1, .1], "half-meter"),
    (20, [.2, 0, 0], [.1, .1, .1], "half-meter"),
])
def test_unusable_geometry_is_rejected(count, minimum, maximum, match):
    with pytest.raises(ValueError, match=match):
        validate_cloud_summary(count, minimum, maximum)
