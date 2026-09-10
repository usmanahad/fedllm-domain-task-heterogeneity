"""Single-process reference runner used on Kaggle and by Flower clients."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import torch

from .aggregation import add_residuals, make_aggregator
from .config import ExperimentConfig
from .diagnostics import geometry_report
from .lora import effective_lora_delta
from .training import (
    build_model_and_tokenizer,
    evaluate_nll,
    get_adapter_state,
    set_adapter_state,
    set_cumulative_base_residual,
    train_client,
    common_checkpoint_gradient,
)
from .types import CanonicalExample, PartitionSpec


def _cell_groups(
    examples: Sequence[CanonicalExample], split: str = "validation"
) -> dict[str, list[CanonicalExample]]:
    grouped: dict[str, list[CanonicalExample]] = defaultdict(list)
    for example in examples:
        if example.split == split:
            grouped[f"{example.domain}/{example.task}"].append(example)
    return dict(grouped)


def run_federated(
    config: ExperimentConfig,
    examples: Sequence[CanonicalExample],
    partitions: PartitionSpec,
    aggregation_method: str,
    output_dir: str | Path,
    device: str | None = None,
) -> dict[str, object]:
    """Run a full-participation sequential simulation with exact accounting."""

    if partitions.num_clients != config.num_clients:
        raise ValueError("Partition/client-count mismatch")
    if aggregation_method in {"ffa", "ffa_lora"} and not config.model.ffa_lora:
        config = replace(config, model=replace(config.model, ffa_lora=True))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_model_and_tokenizer(config.model, device=device)
    global_state = get_adapter_state(model)
    cumulative_residual: dict[str, torch.Tensor] = {}
    applied_residual: dict[str, torch.Tensor] = {}
    lookup = {item.example_id: item for item in examples}
    cells = _cell_groups(examples)
    aggregator = make_aggregator(
        aggregation_method,
        scaling=config.model.scaling,
        rank=config.model.lora_rank,
    )
    logs: list[dict[str, object]] = []

    for round_id in range(1, config.rounds + 1):
        client_updates = []
        global_effective = effective_lora_delta(global_state, config.model.scaling)
        local_effective_updates = []
        true_gradients = []
        diagnostic_round = round_id in config.diagnostic_rounds
        for position, (client_id, ids) in enumerate(sorted(partitions.assignments.items())):
            client_examples = [lookup[example_id] for example_id in ids]
            if diagnostic_round:
                probe = sorted(client_examples, key=lambda item: item.example_id)[
                    : config.train.batch_size
                ]
                true_gradients.append(
                    common_checkpoint_gradient(
                        model,
                        tokenizer,
                        global_state,
                        probe,
                        max_length=config.train.max_length,
                    )
                )
            client_train_config = replace(
                config.train,
                seed=config.seed + (10_000 * round_id) + position,
            )
            result = train_client(
                model,
                tokenizer,
                global_state,
                client_examples,
                client_id,
                round_id,
                config.model,
                client_train_config,
            )
            client_updates.append(result.update)
            local_effective_updates.append(
                {
                    module: value - global_effective[module]
                    for module, value in result.update.effective_delta.items()
                }
            )

        diagnostic_payload: dict[str, object] = {}
        if diagnostic_round:
            probe_cells = {
                cell: sorted(values, key=lambda item: item.example_id)[
                    : config.probe_examples_per_cell
                ]
                for cell, values in cells.items()
            }
            set_adapter_state(model, global_state)
            baseline_probe = {
                cell: evaluate_nll(
                    model,
                    tokenizer,
                    global_state,
                    values,
                    max_length=config.train.max_length,
                    batch_size=config.train.eval_batch_size,
                )
                for cell, values in sorted(probe_cells.items())
            }
            transfer_rows: list[list[float]] = []
            for update in client_updates:
                row = []
                for cell, values in sorted(probe_cells.items()):
                    local_loss = evaluate_nll(
                        model,
                        tokenizer,
                        update.adapter_state,
                        values,
                        max_length=config.train.max_length,
                        batch_size=config.train.eval_batch_size,
                    )
                    row.append(local_loss - baseline_probe[cell])
                transfer_rows.append(row)
            set_adapter_state(model, global_state)
            flattened_transfer = [value for row in transfer_rows for value in row]
            diagnostic_payload = {
                "true_gradient_geometry": geometry_report(
                    true_gradients, [1.0] * len(true_gradients)
                ),
                "functional_transfer": {
                    "client_ids": [update.client_id for update in client_updates],
                    "cells": sorted(probe_cells),
                    "loss_delta_matrix": transfer_rows,
                    "negative_transfer_fraction": (
                        sum(value > 0 for value in flattened_transfer)
                        / max(len(flattened_transfer), 1)
                    ),
                    "baseline_nll": baseline_probe,
                },
            }

        aggregation = aggregator.aggregate(global_state, client_updates)
        global_state = dict(aggregation.adapter_state)
        if aggregation.base_residual:
            cumulative_residual = add_residuals(
                cumulative_residual, dict(aggregation.base_residual)
            )
            applied_residual = set_cumulative_base_residual(
                model, cumulative_residual, applied_residual
            )
        set_adapter_state(model, global_state)

        weights = [float(update.num_examples) for update in client_updates]
        if diagnostic_round:
            normalized = [weight / sum(weights) for weight in weights]
            aggregate_probe_delta = {
                cell: evaluate_nll(
                    model,
                    tokenizer,
                    global_state,
                    values,
                    max_length=config.train.max_length,
                    batch_size=config.train.eval_batch_size,
                )
                - baseline_probe[cell]
                for cell, values in sorted(probe_cells.items())
            }
            weighted_client_delta = {
                cell: sum(
                    normalized[row] * transfer_rows[row][column]
                    for row in range(len(transfer_rows))
                )
                for column, cell in enumerate(sorted(probe_cells))
            }
            diagnostic_payload["functional_transfer"]["aggregate_loss_delta"] = aggregate_probe_delta
            diagnostic_payload["functional_transfer"]["weighted_mean_client_loss_delta"] = weighted_client_delta
            torch.save(
                {
                    update.client_id: dict(update.adapter_state)
                    for update in client_updates
                },
                output_dir / f"client-adapters-round-{round_id:03d}.pt",
            )
        cell_nll = {
            cell: evaluate_nll(
                model,
                tokenizer,
                global_state,
                values,
                max_length=config.train.max_length,
                batch_size=config.train.eval_batch_size,
            )
            for cell, values in sorted(cells.items())
        }
        logs.append(
            {
                "round": round_id,
                "aggregation": aggregation_method,
                "train_loss_by_client": {
                    update.client_id: update.metrics["train_loss"] for update in client_updates
                },
                "validation_nll": cell_nll,
                "geometry": geometry_report(local_effective_updates, weights),
                "aggregation_diagnostics": asdict(aggregation.diagnostics),
                **diagnostic_payload,
            }
        )
        (output_dir / "rounds.json").write_text(
            json.dumps(logs, indent=2, sort_keys=True) + "\n"
        )

    torch.save(global_state, output_dir / "adapter_state.pt")
    torch.save(cumulative_residual, output_dir / "base_residual.pt")
    summary = {
        "experiment": config.name,
        "seed": config.seed,
        "regime": partitions.regime,
        "aggregation": aggregation_method,
        "rounds": config.rounds,
        "example_pool_hash": partitions.example_pool_hash,
        "final_validation_nll": logs[-1]["validation_nll"],
        "artifacts": {
            "adapter_state": "adapter_state.pt",
            "base_residual": "base_residual.pt",
            "rounds": "rounds.json",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def run_reference(
    config: ExperimentConfig,
    examples: Sequence[CanonicalExample],
    arm: str,
    output_dir: str | Path,
    partitions: PartitionSpec | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Run the pretrained-base, centralized, or local-only reference arm."""

    if arm not in {"base", "centralized", "local_only"}:
        raise ValueError("arm must be base, centralized, or local_only")
    if arm == "local_only" and partitions is None:
        raise ValueError("local_only requires a partition specification")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_model_and_tokenizer(config.model, device=device)
    initial = get_adapter_state(model)
    cells = _cell_groups(examples)

    def evaluate(state):
        return {
            cell: evaluate_nll(
                model,
                tokenizer,
                state,
                values,
                max_length=config.train.max_length,
                batch_size=config.train.eval_batch_size,
            )
            for cell, values in sorted(cells.items())
        }

    if arm == "base":
        cell_nll = evaluate(initial)
        result: dict[str, object] = {"arm": arm, "validation_nll": cell_nll}
    elif arm == "centralized":
        training = [item for item in examples if item.split == "train"]
        central_config = replace(
            config.train,
            max_steps=config.rounds * config.train.max_steps * config.num_clients,
        )
        trained = train_client(
            model,
            tokenizer,
            initial,
            training,
            "centralized",
            0,
            config.model,
            central_config,
        )
        torch.save(dict(trained.update.adapter_state), output_dir / "adapter_state.pt")
        result = {
            "arm": arm,
            "steps": central_config.max_steps,
            "train_loss": trained.mean_loss,
            "validation_nll": evaluate(trained.update.adapter_state),
        }
    else:
        assert partitions is not None
        lookup = {item.example_id: item for item in examples}
        local_config = replace(
            config.train,
            max_steps=config.rounds * config.train.max_steps,
        )
        matrix: dict[str, dict[str, float]] = {}
        for position, (client_id, ids) in enumerate(sorted(partitions.assignments.items())):
            local_seeded = replace(local_config, seed=config.seed + position)
            trained = train_client(
                model,
                tokenizer,
                initial,
                [lookup[example_id] for example_id in ids],
                client_id,
                0,
                config.model,
                local_seeded,
            )
            matrix[client_id] = evaluate(trained.update.adapter_state)
        cell_nll = {
            cell: sum(client_scores[cell] for client_scores in matrix.values()) / len(matrix)
            for cell in sorted(cells)
        }
        result = {
            "arm": arm,
            "steps_per_client": local_config.max_steps,
            "validation_nll": cell_nll,
            "client_by_cell_nll": matrix,
        }

    result["experiment"] = config.name
    result["seed"] = config.seed
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
