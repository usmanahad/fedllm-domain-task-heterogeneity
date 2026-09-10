"""Causal validation and guarded weighting for ProToken-like scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class AttributionValidation:
    spearman_correlation: float
    p_value: float
    top_harmful_match: bool
    passed: bool
    minimum_correlation: float


def aggregate_token_attribution(
    token_client_scores: Sequence[Mapping[str, float]],
    client_ids: Sequence[str],
) -> dict[str, float]:
    """Mean normalized contribution per client over attributed output tokens."""

    totals = {client_id: 0.0 for client_id in client_ids}
    used = 0
    for token_scores in token_client_scores:
        vector = np.asarray([max(0.0, token_scores.get(client_id, 0.0)) for client_id in client_ids])
        if vector.sum() <= 0:
            continue
        vector = vector / vector.sum()
        for index, client_id in enumerate(client_ids):
            totals[client_id] += float(vector[index])
        used += 1
    if used:
        totals = {key: value / used for key, value in totals.items()}
    return totals


def validate_attribution_against_harm(
    predicted_harm: Mapping[str, float],
    measured_leave_one_out_harm: Mapping[str, float],
    minimum_correlation: float = 0.5,
    maximum_p_value: float = 0.05,
) -> AttributionValidation:
    clients = sorted(set(predicted_harm).intersection(measured_leave_one_out_harm))
    if len(clients) < 4:
        raise ValueError("At least four shared clients are required for validation")
    predicted = [predicted_harm[client] for client in clients]
    measured = [measured_leave_one_out_harm[client] for client in clients]
    correlation, p_value = spearmanr(predicted, measured)
    correlation = float(correlation) if np.isfinite(correlation) else 0.0
    p_value = float(p_value) if np.isfinite(p_value) else 1.0
    top_match = clients[int(np.argmax(predicted))] == clients[int(np.argmax(measured))]
    return AttributionValidation(
        spearman_correlation=correlation,
        p_value=p_value,
        top_harmful_match=top_match,
        passed=(correlation >= minimum_correlation and p_value <= maximum_p_value),
        minimum_correlation=minimum_correlation,
    )


def attribution_guided_weights(
    predicted_harm: Mapping[str, float],
    temperature: float,
    floor_fraction: float = 0.25,
    validation: AttributionValidation | None = None,
) -> dict[str, float]:
    """Down-weight predicted harm, but only after causal validation passes."""

    if validation is None or not validation.passed:
        raise ValueError("Attribution-guided weighting is gated on passed causal validation")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= floor_fraction <= 1:
        raise ValueError("floor_fraction must be in [0, 1]")
    clients = sorted(predicted_harm)
    harm = np.asarray([predicted_harm[client] for client in clients], dtype=np.float64)
    logits = -(harm - harm.mean()) / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    floor = floor_fraction / len(clients)
    weights = np.maximum(weights, floor)
    weights /= weights.sum()
    return {client: float(weights[index]) for index, client in enumerate(clients)}

