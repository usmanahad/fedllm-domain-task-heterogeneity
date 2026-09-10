from __future__ import annotations

import pytest

from fedllm_heterogeneity.attribution import (
    attribution_guided_weights,
    validate_attribution_against_harm,
)
from fedllm_heterogeneity.evaluation import exact_match, normalized_recovery, token_f1
from fedllm_heterogeneity.stats import equivalence_decision


def test_text_metrics_and_recovery():
    assert exact_match("Hello   world", "hello world") == 1
    assert token_f1("alpha beta", "alpha gamma") == 0.5
    assert normalized_recovery(0.7, 0.5, 0.9) == pytest.approx(0.5)
    assert normalized_recovery(1.5, 2.0, 1.0, higher_is_better=False) == pytest.approx(0.5)


def test_equivalence_uses_lower_confidence_bound():
    decision = equivalence_decision(
        [0.99, 1.00, 1.01, 1.00],
        [1.00, 1.00, 1.00, 1.00],
        margin=-0.05,
        bootstrap_samples=500,
        seed=2,
    )
    assert decision.equivalent


def test_attribution_weighting_is_gated():
    predicted = {f"c{i}": float(i) for i in range(8)}
    measured = {f"c{i}": float(i) for i in range(8)}
    validation = validate_attribution_against_harm(predicted, measured)
    assert validation.passed
    weights = attribution_guided_weights(predicted, 1.0, validation=validation)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["c0"] > weights["c7"]
    with pytest.raises(ValueError):
        attribution_guided_weights(predicted, 1.0, validation=None)

