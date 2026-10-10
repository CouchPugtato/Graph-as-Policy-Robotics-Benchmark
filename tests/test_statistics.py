import pytest

from gap_isaac.statistics import wilson_interval


def test_wilson_interval_for_reproduction_gate() -> None:
    result = wilson_interval(90, 100)
    assert result.rate == 0.9
    assert result.lower == pytest.approx(0.8256, abs=0.0001)
    assert result.upper == pytest.approx(0.9448, abs=0.0001)


@pytest.mark.parametrize("successes,trials", [(-1, 10), (11, 10), (0, 0)])
def test_wilson_interval_rejects_invalid_counts(successes: int, trials: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, trials)
