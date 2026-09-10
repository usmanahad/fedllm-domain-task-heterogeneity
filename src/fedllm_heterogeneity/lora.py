"""LoRA aggregation in factor and induced-weight space."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import torch

from .types import AggregationDiagnostics, AggregationResult


TensorState = Mapping[str, torch.Tensor]
_A_PATTERN = re.compile(r"^(.*)\.lora_A(?:\.[^.]+)?\.weight$")


def lora_factor_pairs(state: TensorState) -> dict[str, tuple[str, str]]:
    """Return module name -> (A key, B key) for a PEFT state dictionary."""

    pairs: dict[str, tuple[str, str]] = {}
    for key in state:
        match = _A_PATTERN.match(key)
        if not match:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.", 1)
        if b_key not in state:
            raise KeyError(f"Missing B factor for {key}; expected {b_key}")
        pairs[match.group(1)] = (key, b_key)
    if not pairs:
        raise ValueError("No LoRA A/B factor pairs found in state dictionary")
    return pairs


def normalized_weights(weights: Sequence[float]) -> torch.Tensor:
    result = torch.as_tensor(weights, dtype=torch.float64)
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError("weights must be a non-empty one-dimensional sequence")
    if torch.any(result < 0) or not torch.isfinite(result).all():
        raise ValueError("weights must be finite and non-negative")
    total = result.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return result / total


def _scale(module: str, scaling: float | Mapping[str, float]) -> float:
    value = scaling[module] if isinstance(scaling, Mapping) else scaling
    if value <= 0:
        raise ValueError(f"LoRA scaling must be positive for {module}")
    return float(value)


def weighted_state_average(
    states: Sequence[TensorState], weights: Sequence[float]
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("At least one client state is required")
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("All client states must contain identical keys")
    norm = normalized_weights(weights)
    result: dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        reference = states[0][key]
        value = sum(
            state[key].detach().to(dtype=torch.float64, device="cpu") * norm[index]
            for index, state in enumerate(states)
        )
        result[key] = value.to(dtype=reference.dtype)
    return result


def effective_lora_delta(
    state: TensorState, scaling: float | Mapping[str, float] = 1.0
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for module, (a_key, b_key) in lora_factor_pairs(state).items():
        a = state[a_key].detach().to(dtype=torch.float64, device="cpu")
        b = state[b_key].detach().to(dtype=torch.float64, device="cpu")
        result[module] = _scale(module, scaling) * (b @ a)
    return result


def weighted_effective_average(
    states: Sequence[TensorState],
    weights: Sequence[float],
    scaling: float | Mapping[str, float] = 1.0,
) -> dict[str, torch.Tensor]:
    norm = normalized_weights(weights)
    deltas = [effective_lora_delta(state, scaling) for state in states]
    modules = set(deltas[0])
    if any(set(delta) != modules for delta in deltas[1:]):
        raise ValueError("All clients must update the same LoRA modules")
    return {
        module: sum(delta[module] * norm[index] for index, delta in enumerate(deltas))
        for module in sorted(modules)
    }


def factor_fedavg(
    states: Sequence[TensorState], weights: Sequence[float]
) -> dict[str, torch.Tensor]:
    """The conventional but generally inexact independent-factor average."""

    return weighted_state_average(states, weights)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).item()
    numerator = torch.linalg.vector_norm(actual - expected).item()
    return float(numerator / max(denominator, 1e-12))


def fedex_lora(
    states: Sequence[TensorState],
    weights: Sequence[float],
    scaling: float | Mapping[str, float] = 1.0,
) -> AggregationResult:
    """Return factor averages plus the exact full-rank FedEx residual.

    Adding ``base_residual[module]`` to the frozen base weight makes the
    resulting effective model exactly equal to the weighted average of client
    effective models, up to floating-point precision.
    """

    adapter = factor_fedavg(states, weights)
    target = weighted_effective_average(states, weights, scaling)
    factor_effective = effective_lora_delta(adapter, scaling)
    residual = {
        module: target[module] - factor_effective[module] for module in target
    }
    errors = {
        module: _relative_error(factor_effective[module], target[module])
        for module in target
    }
    exact = {
        module: factor_effective[module] + residual[module] for module in target
    }
    return AggregationResult(
        adapter_state=adapter,
        base_residual=residual,
        effective_delta=exact,
        diagnostics=AggregationDiagnostics(
            method="fedex_lora",
            client_weights=tuple(float(x) for x in normalized_weights(weights).tolist()),
            factor_average_error=errors,
            reconstruction_error={module: _relative_error(exact[module], target[module]) for module in target},
            notes=("Apply base_residual to the corresponding frozen base weights.",),
        ),
    )


def svd_effective_aggregate(
    states: Sequence[TensorState],
    weights: Sequence[float],
    scaling: float | Mapping[str, float] = 1.0,
    rank: int | None = None,
) -> AggregationResult:
    """Aggregate in weight space and project back into the configured LoRA rank."""

    if not states:
        raise ValueError("At least one state is required")
    pairs = lora_factor_pairs(states[0])
    target = weighted_effective_average(states, weights, scaling)
    adapter = weighted_state_average(states, weights)
    reconstructed: dict[str, torch.Tensor] = {}
    errors: dict[str, float] = {}

    for module, (a_key, b_key) in pairs.items():
        a_template = states[0][a_key]
        b_template = states[0][b_key]
        capacity = min(a_template.shape[0], b_template.shape[1])
        requested_rank = capacity if rank is None else rank
        if requested_rank <= 0 or requested_rank > capacity:
            raise ValueError(
                f"rank for {module} must be in [1, {capacity}], got {requested_rank}"
            )
        matrix = target[module].to(dtype=torch.float64)
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        used = min(requested_rank, singular.numel())
        root = torch.sqrt(singular[:used].clamp_min(0))
        b_small = u[:, :used] * root.unsqueeze(0)
        a_small = root.unsqueeze(1) * vh[:used, :]
        b_small = b_small / _scale(module, scaling)

        a = torch.zeros_like(a_template, dtype=torch.float64, device="cpu")
        b = torch.zeros_like(b_template, dtype=torch.float64, device="cpu")
        a[:used, :] = a_small
        b[:, :used] = b_small
        adapter[a_key] = a.to(dtype=a_template.dtype)
        adapter[b_key] = b.to(dtype=b_template.dtype)
        approximation = _scale(module, scaling) * (b @ a)
        reconstructed[module] = approximation
        errors[module] = _relative_error(approximation, matrix)

    return AggregationResult(
        adapter_state=adapter,
        base_residual={},
        effective_delta=reconstructed,
        diagnostics=AggregationDiagnostics(
            method="svd_effective",
            client_weights=tuple(float(x) for x in normalized_weights(weights).tolist()),
            factor_average_error={},
            reconstruction_error=errors,
            notes=("Reconstruction error is expected when the aggregate rank exceeds the adapter rank.",),
        ),
    )


def validate_ffa_lora(
    states: Sequence[TensorState],
    weights: Sequence[float],
    scaling: float | Mapping[str, float] = 1.0,
    atol: float = 1e-6,
) -> dict[str, float]:
    """Verify that all A factors are shared and factor averaging is exact."""

    pairs = lora_factor_pairs(states[0])
    for module, (a_key, _) in pairs.items():
        reference = states[0][a_key].detach().cpu()
        for index, state in enumerate(states[1:], start=1):
            if not torch.allclose(reference, state[a_key].detach().cpu(), atol=atol, rtol=0):
                raise ValueError(f"Client {index} has a different frozen A factor for {module}")
    averaged = factor_fedavg(states, weights)
    target = weighted_effective_average(states, weights, scaling)
    actual = effective_lora_delta(averaged, scaling)
    return {module: _relative_error(actual[module], target[module]) for module in target}


def apply_base_residual(
    model: torch.nn.Module,
    residual: Mapping[str, torch.Tensor],
) -> None:
    """Apply FedEx residual matrices to matching frozen base-module weights."""

    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, delta in residual.items():
            candidates = [name, name.removeprefix("base_model.model.")]
            target = next((modules[item] for item in candidates if item in modules), None)
            if target is None or not hasattr(target, "weight"):
                raise KeyError(f"Could not resolve base module for residual {name}")
            weight = target.weight
            oriented = delta
            # Transformers GPT-2 uses Conv1D modules whose stored weight is
            # transposed relative to nn.Linear/LoRA's logical delta matrix.
            if tuple(oriented.shape) != tuple(weight.shape):
                if tuple(oriented.T.shape) == tuple(weight.shape):
                    oriented = oriented.T
                else:
                    raise ValueError(
                        f"Residual shape {tuple(delta.shape)} does not match "
                        f"base weight {name} {tuple(weight.shape)}"
                    )
            weight.add_(oriented.to(device=weight.device, dtype=weight.dtype))
