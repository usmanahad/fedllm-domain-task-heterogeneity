"""Public statistical API, named to avoid ambiguity with the stdlib module."""

from __future__ import annotations

from typing import Sequence

from .statistics import (
    ConfidenceInterval,
    EquivalenceDecision,
    noninferiority_decision,
    paired_bootstrap_interval,
)


def equivalence_decision(
    candidate: Sequence[float],
    reference: Sequence[float],
    margin: float = -0.05,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> EquivalenceDecision:
    """Compatibility name for the pre-registered one-sided equivalence test."""

    return noninferiority_decision(
        candidate,
        reference,
        margin=margin,
        confidence=confidence,
        resamples=bootstrap_samples,
        seed=seed,
    )


__all__ = [
    "ConfidenceInterval",
    "EquivalenceDecision",
    "equivalence_decision",
    "noninferiority_decision",
    "paired_bootstrap_interval",
]

