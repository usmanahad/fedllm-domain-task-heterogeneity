"""Geometry, transfer, and token-distribution diagnostics."""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Callable, Mapping, Sequence, TypeVar

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon


T = TypeVar("T")


def flatten_update(update: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not update:
        raise ValueError("Cannot flatten an empty update")
    return torch.cat(
        [update[key].detach().to(dtype=torch.float64, device="cpu").reshape(-1) for key in sorted(update)]
    )


def pairwise_cosines(
    updates: Sequence[Mapping[str, torch.Tensor]],
) -> np.ndarray:
    vectors = [flatten_update(update) for update in updates]
    result: list[float] = []
    for left, right in itertools.combinations(vectors, 2):
        denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        value = 0.0 if denominator == 0 else float(torch.dot(left, right) / denominator)
        result.append(value)
    return np.asarray(result, dtype=np.float64)


def cancellation_ratio(
    updates: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[float]
) -> float:
    if len(updates) != len(weights):
        raise ValueError("updates and weights must have the same length")
    raw = np.asarray(weights, dtype=np.float64)
    raw = raw / raw.sum()
    vectors = [flatten_update(update) for update in updates]
    aggregate = sum(vector * raw[index] for index, vector in enumerate(vectors))
    denominator = sum(
        raw[index] * torch.linalg.vector_norm(vector).item()
        for index, vector in enumerate(vectors)
    )
    return float(torch.linalg.vector_norm(aggregate).item() / max(denominator, 1e-12))


def global_alignment(
    updates: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[float]
) -> list[float]:
    raw = np.asarray(weights, dtype=np.float64)
    raw = raw / raw.sum()
    vectors = [flatten_update(update) for update in updates]
    aggregate = sum(vector * raw[index] for index, vector in enumerate(vectors))
    aggregate_norm = torch.linalg.vector_norm(aggregate)
    result = []
    for vector in vectors:
        denominator = torch.linalg.vector_norm(vector) * aggregate_norm
        result.append(0.0 if denominator == 0 else float(torch.dot(vector, aggregate) / denominator))
    return result


def principal_angles(left: torch.Tensor, right: torch.Tensor, rank: int | None = None) -> np.ndarray:
    """Principal angles in radians between the column spaces of two matrices."""

    left = left.detach().to(dtype=torch.float64, device="cpu")
    right = right.detach().to(dtype=torch.float64, device="cpu")
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Matrices must be 2-D with the same row dimension")
    q_left, _ = torch.linalg.qr(left, mode="reduced")
    q_right, _ = torch.linalg.qr(right, mode="reduced")
    singular = torch.linalg.svdvals(q_left.T @ q_right).clamp(0, 1)
    if rank is not None:
        singular = singular[:rank]
    return torch.arccos(singular).numpy()


def geometry_report(
    updates: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[float]
) -> dict[str, object]:
    cosines = pairwise_cosines(updates)
    norms = [float(torch.linalg.vector_norm(flatten_update(item))) for item in updates]
    return {
        "num_clients": len(updates),
        "pairwise_cosines": cosines.tolist(),
        "mean_cosine": float(cosines.mean()) if cosines.size else None,
        "conflict_fraction": float((cosines < 0).mean()) if cosines.size else None,
        "update_norms": norms,
        "magnitude_ratio": max(norms) / max(min(norms), 1e-12),
        "cancellation_ratio": cancellation_ratio(updates, weights),
        "global_alignment": global_alignment(updates, weights),
    }


def functional_transfer_matrix(
    client_updates: Sequence[T],
    evaluation_cells: Sequence[str],
    apply_update: Callable[[T], None],
    evaluate_cell: Callable[[str], float],
    restore_global: Callable[[], None],
    baseline_losses: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return loss change (post-update minus baseline) for client × cell."""

    if baseline_losses is None:
        restore_global()
        baseline_losses = {cell: float(evaluate_cell(cell)) for cell in evaluation_cells}
    matrix = np.empty((len(client_updates), len(evaluation_cells)), dtype=np.float64)
    for row, update in enumerate(client_updates):
        restore_global()
        apply_update(update)
        for column, cell in enumerate(evaluation_cells):
            matrix[row, column] = float(evaluate_cell(cell)) - baseline_losses[cell]
    restore_global()
    return matrix


def leave_one_out_harm(
    client_updates: Sequence[T],
    aggregate: Callable[[Sequence[T]], T],
    score: Callable[[T], float],
) -> np.ndarray:
    """Positive values mean removing the client improves the score."""

    if len(client_updates) < 2:
        raise ValueError("Leave-one-out analysis needs at least two clients")
    full_score = float(score(aggregate(client_updates)))
    harms = []
    for index in range(len(client_updates)):
        subset = [item for position, item in enumerate(client_updates) if position != index]
        harms.append(float(score(aggregate(subset))) - full_score)
    return np.asarray(harms, dtype=np.float64)


def token_distribution(token_sequences: Sequence[Sequence[int]]) -> Counter[int]:
    counter: Counter[int] = Counter()
    for sequence in token_sequences:
        counter.update(sequence)
    return counter


def token_js_divergence(left: Counter[int], right: Counter[int]) -> float:
    vocabulary = sorted(set(left).union(right))
    if not vocabulary:
        return 0.0
    p = np.asarray([left[token] for token in vocabulary], dtype=np.float64)
    q = np.asarray([right[token] for token in vocabulary], dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q, base=2.0) ** 2)


def shared_token_mass(left: Counter[int], right: Counter[int]) -> float:
    """Mass carried by tokens that occur at least once in both samples.

    This support-based diagnostic is retained for compatibility with the
    original data manifest.  For distributional similarity, prefer
    :func:`weighted_token_overlap`, which also accounts for frequency mismatch.
    """
    if not left or not right:
        return 0.0
    shared = set(left).intersection(right)
    left_mass = sum(left[token] for token in shared) / sum(left.values())
    right_mass = sum(right[token] for token in shared) / sum(right.values())
    return float((left_mass + right_mass) / 2)


def weighted_token_overlap(left: Counter[int], right: Counter[int]) -> float:
    """Overlap coefficient between normalized token-frequency distributions.

    The result is ``sum_t min(p_left(t), p_right(t))`` and equals ``1 - TV``.
    It is one only for identical distributions and zero for disjoint support.
    """

    if not left or not right:
        return 0.0
    left_total = float(sum(left.values()))
    right_total = float(sum(right.values()))
    return float(
        sum(
            min(left[token] / left_total, right[token] / right_total)
            for token in set(left).union(right)
        )
    )


def vocabulary_jaccard(left: Counter[int], right: Counter[int]) -> float:
    """Jaccard similarity of the observed token vocabularies."""

    union = set(left).union(right)
    if not union:
        return 1.0
    return float(len(set(left).intersection(right)) / len(union))


def ubiquitous_token_mass(distributions: Mapping[str, Counter[int]]) -> float:
    """Mean probability mass on tokens observed in every supplied group."""

    nonempty = {key: value for key, value in distributions.items() if value}
    if not nonempty:
        return 0.0
    shared = set.intersection(*(set(value) for value in nonempty.values()))
    return float(
        np.mean(
            [
                sum(counter[token] for token in shared) / sum(counter.values())
                for counter in nonempty.values()
            ]
        )
    )


def token_label_mutual_information(
    distributions: Mapping[str, Counter[int]],
) -> tuple[float, dict[int, float]]:
    """Mutual information in bits between a token and an equal-prior label.

    Each group's token distribution is normalized independently and labels are
    assigned a uniform prior.  This prevents longer-domain examples from
    receiving a larger prior merely because they contain more tokens.  The
    second return value gives each token's signed contribution to total MI.
    """

    nonempty = {key: value for key, value in distributions.items() if value}
    if len(nonempty) < 2:
        return 0.0, {}
    labels = sorted(nonempty)
    prior = 1.0 / len(labels)
    conditional = {
        label: {token: count / sum(nonempty[label].values()) for token, count in nonempty[label].items()}
        for label in labels
    }
    marginal: Counter[int] = Counter()
    for label in labels:
        for token, probability in conditional[label].items():
            marginal[token] += prior * probability
    contributions: dict[int, float] = {}
    for token, token_probability in marginal.items():
        contribution = 0.0
        for label in labels:
            probability = conditional[label].get(token, 0.0)
            if probability > 0:
                joint = prior * probability
                contribution += joint * np.log2(probability / token_probability)
        contributions[token] = float(contribution)
    return float(sum(contributions.values())), contributions


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear centered-kernel alignment for two activation matrices."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Activation matrices must be 2-D with equal row counts")
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(left.T @ right, ord="fro") ** 2
    denominator = np.linalg.norm(left.T @ left, ord="fro") * np.linalg.norm(
        right.T @ right, ord="fro"
    )
    return 0.0 if denominator == 0 else float(cross / denominator)
