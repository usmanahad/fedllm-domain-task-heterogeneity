"""Pre-registered uncertainty and equivalence calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class EquivalenceDecision:
    difference: ConfidenceInterval
    lower_margin: float
    equivalent: bool


def paired_bootstrap_interval(
    left: Sequence[float],
    right: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 42,
) -> ConfidenceInterval:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("left and right must be paired one-dimensional arrays")
    if left_array.size < 2:
        raise ValueError("At least two paired observations are required")
    differences = left_array - right_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    bootstrap = differences[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return ConfidenceInterval(
        estimate=float(differences.mean()),
        lower=float(np.quantile(bootstrap, alpha)),
        upper=float(np.quantile(bootstrap, 1 - alpha)),
        confidence=confidence,
    )


def noninferiority_decision(
    candidate: Sequence[float],
    reference: Sequence[float],
    margin: float = -0.05,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 42,
) -> EquivalenceDecision:
    interval = paired_bootstrap_interval(
        candidate, reference, confidence=confidence, resamples=resamples, seed=seed
    )
    return EquivalenceDecision(
        difference=interval,
        lower_margin=margin,
        equivalent=interval.lower > margin,
    )
