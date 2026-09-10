"""Uniform aggregation interface used by local and Flower simulations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch

from .lora import (
    TensorState,
    effective_lora_delta,
    factor_fedavg,
    fedex_lora,
    normalized_weights,
    svd_effective_aggregate,
    validate_ffa_lora,
)
from .types import AggregationDiagnostics, AggregationResult, ClientUpdate


class Aggregator(ABC):
    """Aggregate complete client adapter states from a shared checkpoint."""

    name: str

    def __init__(self, scaling: float = 1.0) -> None:
        self.scaling = scaling

    @abstractmethod
    def aggregate(
        self, global_state: TensorState, updates: Sequence[ClientUpdate]
    ) -> AggregationResult: ...

    @staticmethod
    def _states_and_weights(
        updates: Sequence[ClientUpdate],
    ) -> tuple[list[TensorState], list[float]]:
        if not updates:
            raise ValueError("No client updates to aggregate")
        states = [update.adapter_state for update in updates]
        # Preserve standard FedAvg semantics. Target-token weighting would let
        # long code responses dominate short finance labels in the natural arm.
        weights = [float(update.num_examples) for update in updates]
        return states, weights


class FactorFedAvgAggregator(Aggregator):
    name = "factor_fedavg"

    def aggregate(
        self, global_state: TensorState, updates: Sequence[ClientUpdate]
    ) -> AggregationResult:
        del global_state
        states, weights = self._states_and_weights(updates)
        adapter = factor_fedavg(states, weights)
        effective = effective_lora_delta(adapter, self.scaling)
        return AggregationResult(
            adapter_state=adapter,
            base_residual={},
            effective_delta=effective,
            diagnostics=AggregationDiagnostics(
                method=self.name,
                client_weights=tuple(normalized_weights(weights).tolist()),
                notes=("Independent A/B averaging is intentionally retained as a diagnostic baseline.",),
            ),
        )


class FFALoRAAggregator(FactorFedAvgAggregator):
    name = "ffa_lora"

    def aggregate(
        self, global_state: TensorState, updates: Sequence[ClientUpdate]
    ) -> AggregationResult:
        del global_state
        states, weights = self._states_and_weights(updates)
        errors = validate_ffa_lora(states, weights, self.scaling)
        adapter = factor_fedavg(states, weights)
        effective = effective_lora_delta(adapter, self.scaling)
        return AggregationResult(
            adapter_state=adapter,
            base_residual={},
            effective_delta=effective,
            diagnostics=AggregationDiagnostics(
                method=self.name,
                client_weights=tuple(normalized_weights(weights).tolist()),
                reconstruction_error=errors,
                notes=("All clients were verified to share the same frozen A factors.",),
            ),
        )


class FedExLoRAAggregator(Aggregator):
    name = "fedex_lora"

    def aggregate(
        self, global_state: TensorState, updates: Sequence[ClientUpdate]
    ) -> AggregationResult:
        del global_state
        states, weights = self._states_and_weights(updates)
        return fedex_lora(states, weights, self.scaling)


class SVDEffectiveAggregator(Aggregator):
    name = "svd_effective"

    def __init__(self, scaling: float = 1.0, rank: int | None = None) -> None:
        super().__init__(scaling)
        self.rank = rank

    def aggregate(
        self, global_state: TensorState, updates: Sequence[ClientUpdate]
    ) -> AggregationResult:
        del global_state
        states, weights = self._states_and_weights(updates)
        return svd_effective_aggregate(states, weights, self.scaling, self.rank)


def make_aggregator(name: str, scaling: float, rank: int | None = None) -> Aggregator:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"fedavg", "factor_fedavg"}:
        return FactorFedAvgAggregator(scaling)
    if normalized in {"ffa", "ffa_lora"}:
        return FFALoRAAggregator(scaling)
    if normalized in {"fedex", "fedex_lora"}:
        return FedExLoRAAggregator(scaling)
    if normalized in {"flora", "svd", "svd_effective"}:
        return SVDEffectiveAggregator(scaling, rank=rank)
    raise ValueError(f"Unknown aggregation method: {name}")


def client_update_from_state(
    client_id: str,
    round_id: int,
    state: TensorState,
    num_examples: int,
    num_input_tokens: int,
    num_target_tokens: int,
    scaling: float,
    metrics: dict[str, float] | None = None,
) -> ClientUpdate:
    return ClientUpdate(
        round_id=round_id,
        client_id=client_id,
        num_examples=num_examples,
        num_input_tokens=num_input_tokens,
        num_target_tokens=num_target_tokens,
        adapter_state=state,
        effective_delta=effective_lora_delta(state, scaling),
        metrics=metrics or {},
    )


def add_residuals(
    cumulative: dict[str, torch.Tensor], increment: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    result = {key: value.detach().cpu().clone() for key, value in cumulative.items()}
    for key, value in increment.items():
        result[key] = result.get(key, torch.zeros_like(value)) + value.detach().cpu()
    return result
