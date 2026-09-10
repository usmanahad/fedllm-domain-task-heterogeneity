from __future__ import annotations

import pytest
import torch

from fedllm_heterogeneity.lora import (
    apply_base_residual,
    effective_lora_delta,
    factor_fedavg,
    fedex_lora,
    svd_effective_aggregate,
    validate_ffa_lora,
    weighted_effective_average,
)


def random_states(seed=7, clients=3, rank=2):
    torch.manual_seed(seed)
    return [
        {
            "layer.lora_A.default.weight": torch.randn(rank, 5),
            "layer.lora_B.default.weight": torch.randn(4, rank),
        }
        for _ in range(clients)
    ]


def relative(actual, expected):
    return torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)


def test_factor_average_is_generally_inexact_but_fedex_is_exact():
    states = random_states()
    target = weighted_effective_average(states, [1, 2, 3])["layer"]
    naive = effective_lora_delta(factor_fedavg(states, [1, 2, 3]))["layer"]
    exact = fedex_lora(states, [1, 2, 3]).effective_delta["layer"]
    assert relative(naive, target) > 1e-3
    assert relative(exact, target) < 1e-10


def test_ffa_is_exact_and_rejects_changed_a():
    states = random_states()
    shared_a = states[0]["layer.lora_A.default.weight"]
    for state in states:
        state["layer.lora_A.default.weight"] = shared_a.clone()
    errors = validate_ffa_lora(states, [1, 1, 1])
    assert errors["layer"] < 1e-6
    states[1]["layer.lora_A.default.weight"][0, 0] += 1
    with pytest.raises(ValueError):
        validate_ffa_lora(states, [1, 1, 1])


def test_svd_aggregation_reports_rank_loss():
    states = random_states(clients=4, rank=2)
    result = svd_effective_aggregate(states, [1, 1, 1, 1], rank=2)
    assert 0 <= result.diagnostics.reconstruction_error["layer"] <= 1
    assert result.adapter_state["layer.lora_A.default.weight"].shape == (2, 5)


def test_base_residual_accepts_conv1d_transposed_storage():
    class TransposedLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(5, 4), requires_grad=False)

    model = torch.nn.Module()
    model.add_module("layer", TransposedLayer())
    delta = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    apply_base_residual(model, {"layer": delta})
    assert torch.equal(model.layer.weight, delta.T)
