"""Flower ClientApp/ServerApp for the unified cross-domain federation.

The wire payload contains the adapter plus the cumulative FedEx base residual.
This is necessary in simulation because every client must train from the exact
same effective global model. Factor/SVD aggregation simply carries zero
residuals.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .aggregation import add_residuals, make_aggregator
from .config import load_config
from .data import read_examples, stable_hash
from .lora import effective_lora_delta
from .partitioning import read_partitions
from .training import (
    build_model_and_tokenizer,
    evaluate_nll,
    get_adapter_state,
    set_adapter_state,
    set_cumulative_base_residual,
    train_client,
)
from .types import ClientUpdate


class StateCodec:
    """Deterministic ndarray codec for adapters and cumulative residuals."""

    def __init__(
        self,
        adapter_template: Mapping[str, torch.Tensor],
        residual_template: Mapping[str, torch.Tensor],
    ) -> None:
        self.adapter_keys = tuple(sorted(adapter_template))
        self.residual_keys = tuple(sorted(residual_template))
        self.adapter_shapes = {key: tuple(adapter_template[key].shape) for key in self.adapter_keys}
        self.residual_shapes = {key: tuple(residual_template[key].shape) for key in self.residual_keys}

    def encode(
        self,
        adapter: Mapping[str, torch.Tensor],
        residual: Mapping[str, torch.Tensor],
    ) -> list[np.ndarray]:
        arrays = [adapter[key].detach().cpu().numpy() for key in self.adapter_keys]
        arrays.extend(
            residual.get(key, torch.zeros(self.residual_shapes[key])).detach().cpu().numpy()
            for key in self.residual_keys
        )
        return arrays

    def decode(
        self, arrays: Sequence[np.ndarray]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        expected = len(self.adapter_keys) + len(self.residual_keys)
        if len(arrays) != expected:
            raise ValueError(f"Expected {expected} arrays, received {len(arrays)}")
        adapter: dict[str, torch.Tensor] = {}
        residual: dict[str, torch.Tensor] = {}
        for index, key in enumerate(self.adapter_keys):
            value = torch.from_numpy(np.asarray(arrays[index])).clone()
            if tuple(value.shape) != self.adapter_shapes[key]:
                raise ValueError(f"Shape mismatch for {key}")
            adapter[key] = value
        offset = len(self.adapter_keys)
        for index, key in enumerate(self.residual_keys):
            value = torch.from_numpy(np.asarray(arrays[offset + index])).clone()
            if tuple(value.shape) != self.residual_shapes[key]:
                raise ValueError(f"Shape mismatch for residual {key}")
            residual[key] = value
        return adapter, residual


def codec_from_adapter(adapter: Mapping[str, torch.Tensor], scaling: float) -> StateCodec:
    effective = effective_lora_delta(adapter, scaling)
    residual = {key: torch.zeros_like(value) for key, value in effective.items()}
    return StateCodec(adapter, residual)


def _run_value(context, key: str) -> str:
    value = context.run_config.get(key)
    if value is None:
        raise KeyError(f"Missing Flower run configuration: {key}")
    return str(value)


def _apply_run_seed(config, context):
    if "seed" not in context.run_config:
        return config
    seed = int(context.run_config["seed"])
    return replace(config, seed=seed, train=replace(config.train, seed=seed))


try:  # The core package remains importable without the optional Flower extra.
    from flwr.client import ClientApp, NumPyClient
    from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
except ImportError:  # pragma: no cover - exercised only without optional dependency
    client_app = None
    server_app = None
else:

    class UnifiedFlowerClient(NumPyClient):
        def __init__(self, context: Context) -> None:
            config_path = _run_value(context, "experiment-config")
            examples_path = _run_value(context, "examples")
            partitions_path = _run_value(context, "partitions")
            aggregation = _run_value(context, "aggregation")
            self.config = _apply_run_seed(load_config(config_path), context)
            if aggregation in {"ffa", "ffa_lora"}:
                self.config = replace(
                    self.config,
                    model=replace(self.config.model, ffa_lora=True),
                )
            partition_id = int(context.node_config["partition-id"])
            self.client_id = f"client_{partition_id:02d}"
            all_examples = read_examples(examples_path)
            partitions = read_partitions(partitions_path)
            lookup = {item.example_id: item for item in all_examples}
            self.training_examples = [
                lookup[example_id] for example_id in partitions.assignments[self.client_id]
            ]
            self.validation_examples = [
                item for item in all_examples if item.split == "validation"
            ]
            self.model, self.tokenizer = build_model_and_tokenizer(self.config.model)
            adapter = get_adapter_state(self.model)
            self.codec = codec_from_adapter(adapter, self.config.model.scaling)
            self.adapter = adapter
            self.residual = {
                key: torch.zeros(shape) for key, shape in self.codec.residual_shapes.items()
            }
            self.applied_residual: dict[str, torch.Tensor] = {}

        def _set_payload(self, parameters: Sequence[np.ndarray]) -> None:
            adapter, residual = self.codec.decode(parameters)
            self.applied_residual = set_cumulative_base_residual(
                self.model, residual, self.applied_residual
            )
            self.adapter = adapter
            self.residual = residual
            set_adapter_state(self.model, self.adapter)

        def get_parameters(self, config):
            del config
            return self.codec.encode(self.adapter, self.residual)

        def fit(self, parameters, config):
            self._set_payload(parameters)
            round_id = int(config.get("server_round", 1))
            offset = int(stable_hash(self.config.seed, self.client_id, length=8), 16)
            train_config = replace(self.config.train, seed=self.config.seed + offset + round_id)
            result = train_client(
                self.model,
                self.tokenizer,
                self.adapter,
                self.training_examples,
                self.client_id,
                round_id,
                self.config.model,
                train_config,
            )
            self.adapter = dict(result.update.adapter_state)
            metrics = {
                "client_id": self.client_id,
                "input_tokens": result.update.num_input_tokens,
                "target_tokens": result.update.num_target_tokens,
                "train_loss": result.mean_loss,
            }
            return (
                self.codec.encode(self.adapter, self.residual),
                result.update.num_examples,
                metrics,
            )

        def evaluate(self, parameters, config):
            del config
            self._set_payload(parameters)
            loss = evaluate_nll(
                self.model,
                self.tokenizer,
                self.adapter,
                self.validation_examples,
                max_length=self.config.train.max_length,
                batch_size=self.config.train.eval_batch_size,
            )
            return loss, len(self.validation_examples), {"validation_nll": loss}


    class ExactLoRAStrategy(FedAvg):
        def __init__(self, context: Context, **kwargs) -> None:
            self.config = _apply_run_seed(
                load_config(_run_value(context, "experiment-config")), context
            )
            self.aggregation_name = _run_value(context, "aggregation")
            if self.aggregation_name in {"ffa", "ffa_lora"}:
                self.config = replace(
                    self.config,
                    model=replace(self.config.model, ffa_lora=True),
                )
            model, _ = build_model_and_tokenizer(self.config.model, device="cpu")
            adapter = get_adapter_state(model)
            self.codec = codec_from_adapter(adapter, self.config.model.scaling)
            self.adapter = adapter
            self.cumulative_residual = {
                key: torch.zeros(shape) for key, shape in self.codec.residual_shapes.items()
            }
            self.output = Path(_run_value(context, "output"))
            self.output.mkdir(parents=True, exist_ok=True)
            initial = ndarrays_to_parameters(
                self.codec.encode(self.adapter, self.cumulative_residual)
            )
            super().__init__(initial_parameters=initial, **kwargs)

        def configure_fit(self, server_round, parameters, client_manager):
            instructions = super().configure_fit(server_round, parameters, client_manager)
            for _, fit_ins in instructions:
                fit_ins.config["server_round"] = server_round
            return instructions

        def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None, {}
            updates = []
            for proxy, fit_res in results:
                arrays = parameters_to_ndarrays(fit_res.parameters)
                adapter, _ = self.codec.decode(arrays)
                metrics = dict(fit_res.metrics)
                updates.append(
                    ClientUpdate(
                        round_id=server_round,
                        client_id=str(metrics.get("client_id", proxy.cid)),
                        num_examples=fit_res.num_examples,
                        num_input_tokens=int(metrics.get("input_tokens", 0)),
                        num_target_tokens=int(metrics.get("target_tokens", 0)),
                        adapter_state=adapter,
                        effective_delta=effective_lora_delta(adapter, self.config.model.scaling),
                        metrics={"train_loss": float(metrics.get("train_loss", float("nan")))},
                    )
                )
            aggregator = make_aggregator(
                self.aggregation_name,
                scaling=self.config.model.scaling,
                rank=self.config.model.lora_rank,
            )
            aggregate = aggregator.aggregate(self.adapter, updates)
            self.adapter = dict(aggregate.adapter_state)
            if aggregate.base_residual:
                self.cumulative_residual = add_residuals(
                    self.cumulative_residual, dict(aggregate.base_residual)
                )
            payload = ndarrays_to_parameters(
                self.codec.encode(self.adapter, self.cumulative_residual)
            )
            record = {
                "round": server_round,
                "aggregation": self.aggregation_name,
                "num_clients": len(updates),
                "failures": len(failures),
                "diagnostics": asdict(aggregate.diagnostics),
            }
            (self.output / f"aggregation-round-{server_round:03d}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n"
            )
            if server_round == self.config.rounds:
                torch.save(self.adapter, self.output / "adapter_state.pt")
                torch.save(self.cumulative_residual, self.output / "base_residual.pt")
                torch.save(
                    {
                        update.client_id: dict(update.adapter_state)
                        for update in updates
                    },
                    self.output / f"client-adapters-round-{server_round:03d}.pt",
                )
            return payload, {"num_aggregated_clients": len(updates)}


    def client_fn(context: Context):
        return UnifiedFlowerClient(context).to_client()


    def server_fn(context: Context) -> ServerAppComponents:
        config = _apply_run_seed(
            load_config(_run_value(context, "experiment-config")), context
        )
        strategy = ExactLoRAStrategy(
            context,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=config.num_clients,
            min_available_clients=config.num_clients,
        )
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=config.rounds),
        )


    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)
