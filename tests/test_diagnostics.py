from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from fedllm_heterogeneity.diagnostics import (
    cancellation_ratio,
    geometry_report,
    linear_cka,
    shared_token_mass,
    token_label_mutual_information,
    token_js_divergence,
    ubiquitous_token_mass,
    vocabulary_jaccard,
    weighted_token_overlap,
)


def test_known_conflict_geometry():
    updates = [
        {"x": torch.tensor([1.0, 0.0])},
        {"x": torch.tensor([-1.0, 0.0])},
        {"x": torch.tensor([0.0, 1.0])},
    ]
    report = geometry_report(updates, [1, 1, 1])
    assert report["conflict_fraction"] == 1 / 3
    assert 0 < cancellation_ratio(updates, [1, 1, 1]) < 1


def test_token_statistics():
    same = Counter({1: 4, 2: 2})
    other = Counter({1: 2, 3: 4})
    assert token_js_divergence(same, same) == 0
    assert 0 < token_js_divergence(same, other) <= 1
    assert 0 < shared_token_mass(same, other) < 1
    assert weighted_token_overlap(same, same) == 1
    assert 0 < weighted_token_overlap(same, other) < 1
    assert vocabulary_jaccard(same, same) == 1
    assert vocabulary_jaccard(Counter(), Counter()) == 1


def test_domain_information_and_ubiquitous_mass():
    shared = {"a": Counter({1: 5, 2: 5}), "b": Counter({1: 5, 2: 5})}
    separated = {"a": Counter({1: 10}), "b": Counter({2: 10})}
    shared_mi, _ = token_label_mutual_information(shared)
    separated_mi, contributions = token_label_mutual_information(separated)
    assert np.isclose(shared_mi, 0)
    assert np.isclose(separated_mi, 1)
    assert np.isclose(sum(contributions.values()), separated_mi)
    assert ubiquitous_token_mass(shared) == 1
    assert ubiquitous_token_mass(separated) == 0


def test_linear_cka_identity_and_independent_shape_check():
    matrix = np.arange(24, dtype=float).reshape(6, 4)
    assert np.isclose(linear_cka(matrix, matrix), 1)
    with np.testing.assert_raises(ValueError):
        linear_cka(matrix, matrix[:-1])
