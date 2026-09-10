from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from fedllm_heterogeneity.diagnostics import (
    cancellation_ratio,
    geometry_report,
    shared_token_mass,
    token_js_divergence,
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

