"""Benchmark summary statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProportionInterval:
    successes: int
    trials: int
    rate: float
    lower: float
    upper: float


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> ProportionInterval:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")

    rate = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (rate + z2 / (2 * trials)) / denominator
    margin = z * math.sqrt((rate * (1 - rate) + z2 / (4 * trials)) / trials) / denominator
    return ProportionInterval(successes, trials, rate, center - margin, center + margin)
