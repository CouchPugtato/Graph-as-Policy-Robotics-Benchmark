import pytest

from gap_isaac.sam3_smoke import select_candidate


def detection(label, score, box=None):
    return {"label": label, "score": score,
            "box": box or {"x1": 370, "y1": 318, "x2": 386, "y2": 341}}


def test_select_exact_blue_label_not_combined_label():
    report = {"detections": [detection("blue cylinder red cylinder", 0.9),
                             detection("blue cylinder", 0.5),
                             detection("blue cylinder", 0.3)]}
    index, selected = select_candidate(report)
    assert index == 1
    assert selected["score"] == 0.5


def test_no_target_is_not_silently_replaced():
    with pytest.raises(ValueError, match="No exact"):
        select_candidate({"detections": [detection("red cylinder", 0.9)]})


def test_select_basket_without_confusing_target():
    report = {"detections": [detection("blue cylinder", .5), detection("green open box", .8)]}
    index, _ = select_candidate(report, "green open box")
    assert index == 1


@pytest.mark.parametrize("box", [
    {"x1": float("nan"), "y1": 318, "x2": 386, "y2": 341},
    {"x1": 370, "y1": 318, "x2": 369, "y2": 341},
    {"x1": 370, "y1": 318, "x2": 641, "y2": 341},
])
def test_invalid_box_rejected(box):
    with pytest.raises(ValueError):
        select_candidate({"detections": [detection("blue cylinder", 0.5, box)]})
