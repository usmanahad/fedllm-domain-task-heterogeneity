"""Single-process reference runner used on Kaggle and by Flower clients."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import torch

from .aggregation import add_residuals, make_aggregator
from .config import ExperimentConfig
from .diagnostics import geometry_report
from .lora import effective_lora_delta
from .partitioning import example_pool_hash
from .training import (
    build_model_and_tokenizer,
    evaluate_nll,
    get_adapter_state,
    set_adapter_state,
    set_cumulative_base_residual,
    seed_everything,
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


def _leave_one_out_payload(
    client_ids: Sequence[str],
    cells: Sequence[str],
    full_aggregate_nll: dict[str, float],
    without_client_nll: Sequence[Sequence[float]],
) -> dict[str, object]:
    """Summarize loss-based leave-one-client-out effects with an explicit sign."""

    if len(client_ids) != len(without_client_nll):
        raise ValueError("Each client must have one leave-one-out result row")
    if not cells:
        raise ValueError("Leave-one-out analysis needs at least one evaluation cell")
    if set(cells) != set(full_aggregate_nll):
        raise ValueError("Full-aggregate NLL keys must match the evaluation cells")
    deltas = []
    for row in without_client_nll:
        if len(row) != len(cells):
            raise ValueError("Each leave-one-out row must contain every evaluation cell")
        deltas.append(
            [value - full_aggregate_nll[cell] for value, cell in zip(row, cells)]
        )
    macro_deltas = [sum(row) / len(row) for row in deltas]
    return {
        "definition": (
            "loss_delta_without_client_minus_full < 0 means removing the client "
            "improves NLL; client_harm_score > 0 therefore means harmful"
        ),
        "client_ids": list(client_ids),
        "cells": list(cells),
        "full_aggregate_nll": full_aggregate_nll,
        "without_client_nll_matrix": [list(row) for row in without_client_nll],
        "loss_delta_without_client_minus_full_matrix": deltas,
        "macro_loss_delta_without_client_minus_full": macro_deltas,
        "client_harm_score": [-value for value in macro_deltas],
    }


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
    seed_everything(config.seed)
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
    run_started = time.perf_counter()

    for round_id in range(1, config.rounds + 1):
        round_started = time.perf_counter()
        timing = {
            "true_gradients": 0.0,
            "client_training": 0.0,
            "pre_aggregation_transfer": 0.0,
            "aggregation": 0.0,
            "post_aggregation_diagnostics": 0.0,
            "validation": 0.0,
        }
        client_updates = []
        global_effective = effective_lora_delta(global_state, config.model.scaling)
        local_effective_updates = []
        true_gradients = []
        leave_one_out_round = round_id in config.leave_one_out_rounds
        diagnostic_round = round_id in set(config.diagnostic_rounds).union(
            config.leave_one_out_rounds
        )
        for position, (client_id, ids) in enumerate(sorted(partitions.assignments.items())):
            client_examples = [lookup[example_id] for example_id in ids]
            if diagnostic_round:
                probe = sorted(client_examples, key=lambda item: item.example_id)[
                    : config.gradient_probe_examples_per_client
                ]
                gradient_started = time.perf_counter()
                true_gradients.append(
                    common_checkpoint_gradient(
                        model,
                        tokenizer,
                        global_state,
                        probe,
                        max_length=config.train.max_length,
                        microbatch_size=config.train.eval_batch_size,
                    )
                )
                timing["true_gradients"] += time.perf_counter() - gradient_started
            client_train_config = replace(
                config.train,
                seed=config.seed + (10_000 * round_id) + position,
            )
            training_started = time.perf_counter()
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
            timing["client_training"] += time.perf_counter() - training_started
            client_updates.append(result.update)
            local_effective_updates.append(
                {
                    module: value - global_effective[module]
                    for module, value in result.update.effective_delta.items()
                }
            )

        diagnostic_payload: dict[str, object] = {}
        if diagnostic_round:
            transfer_started = time.perf_counter()
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
            timing["pre_aggregation_transfer"] = (
                time.perf_counter() - transfer_started
            )

        previous_global_state = global_state
        previous_cumulative_residual = dict(cumulative_residual)
        aggregation_started = time.perf_counter()
        aggregation = aggregator.aggregate(previous_global_state, client_updates)
        global_state = dict(aggregation.adapter_state)
        if aggregation.base_residual:
            cumulative_residual = add_residuals(
                cumulative_residual, dict(aggregation.base_residual)
            )
            applied_residual = set_cumulative_base_residual(
                model, cumulative_residual, applied_residual
            )
        set_adapter_state(model, global_state)
        timing["aggregation"] = time.perf_counter() - aggregation_started

        weights = [float(update.num_examples) for update in client_updates]
        if diagnostic_round:
            post_diagnostic_started = time.perf_counter()
            normalized = [weight / sum(weights) for weight in weights]
            aggregate_probe_nll = {
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
            aggregate_probe_delta = {
                cell: aggregate_probe_nll[cell] - baseline_probe[cell]
                for cell in sorted(probe_cells)
            }
            weighted_client_delta = {
                cell: sum(
                    normalized[row] * transfer_rows[row][column]
                    for row in range(len(transfer_rows))
                )
                for column, cell in enumerate(sorted(probe_cells))
            }
            diagnostic_payload["functional_transfer"]["aggregate_loss_delta"] = aggregate_probe_delta
            diagnostic_payload["functional_transfer"]["aggregate_nll"] = aggregate_probe_nll
            diagnostic_payload["functional_transfer"]["weighted_mean_client_loss_delta"] = weighted_client_delta
            if leave_one_out_round:
                without_client_nll: list[list[float]] = []
                sorted_cells = sorted(probe_cells)
                for removed_index, _ in enumerate(client_updates):
                    retained = [
                        update
                        for index, update in enumerate(client_updates)
                        if index != removed_index
                    ]
                    candidate = aggregator.aggregate(previous_global_state, retained)
                    candidate_cumulative = add_residuals(
                        previous_cumulative_residual,
                        dict(candidate.base_residual),
                    )
                    candidate_applied = set_cumulative_base_residual(
                        model,
                        candidate_cumulative,
                        applied_residual,
                    )
                    candidate_state = dict(candidate.adapter_state)
                    row = [
                        evaluate_nll(
                            model,
                            tokenizer,
                            candidate_state,
                            probe_cells[cell],
                            max_length=config.train.max_length,
                            batch_size=config.train.eval_batch_size,
                        )
                        for cell in sorted_cells
                    ]
                    without_client_nll.append(row)
                    applied_residual = set_cumulative_base_residual(
                        model,
                        cumulative_residual,
                        candidate_applied,
                    )
                    set_adapter_state(model, global_state)
                diagnostic_payload["leave_one_out_aggregation"] = _leave_one_out_payload(
                    [update.client_id for update in client_updates],
                    sorted_cells,
                    aggregate_probe_nll,
                    without_client_nll,
                )
            if config.save_checkpoints:
                torch.save(
                    {
                        update.client_id: dict(update.adapter_state)
                        for update in client_updates
                    },
                    output_dir / f"client-adapters-round-{round_id:03d}.pt",
                )
            timing["post_aggregation_diagnostics"] = (
                time.perf_counter() - post_diagnostic_started
            )
        validation_started = time.perf_counter()
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
        timing["validation"] = time.perf_counter() - validation_started
        timing["total"] = time.perf_counter() - round_started
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
                "elapsed_seconds": timing,
                **diagnostic_payload,
            }
        )
        (output_dir / "rounds.json").write_text(
            json.dumps(logs, indent=2, sort_keys=True) + "\n"
        )

    artifacts = {"rounds": "rounds.json"}
    if config.save_checkpoints:
        torch.save(global_state, output_dir / "adapter_state.pt")
        torch.save(cumulative_residual, output_dir / "base_residual.pt")
        artifacts.update(
            {
                "adapter_state": "adapter_state.pt",
                "base_residual": "base_residual.pt",
            }
        )
    summary = {
        "experiment": config.name,
        "data_version": config.data_version,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "seed": config.seed,
        "regime": partitions.regime,
        "aggregation": aggregation_method,
        "rounds": config.rounds,
        "num_clients": partitions.num_clients,
        "partition_seed": partitions.seed,
        "client_cell_weights": partitions.cell_weights,
        "example_pool_hash": partitions.example_pool_hash,
        "final_validation_nll": logs[-1]["validation_nll"],
        "elapsed_seconds": {
            "total": time.perf_counter() - run_started,
            "rounds": [item["elapsed_seconds"] for item in logs],
        },
        "artifacts": artifacts,
        "reproducibility": {
            "optimization_seed": config.seed,
            "client_rng_reset": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
    }
    if "leave_one_out_aggregation" in logs[-1]:
        summary["final_leave_one_out_aggregation"] = logs[-1][
            "leave_one_out_aggregation"
        ]
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
    run_started = time.perf_counter()
    seed_everything(config.seed)
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
        if config.save_checkpoints:
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
    result["data_version"] = config.data_version
    result["model_id"] = config.model.model_id
    result["model_revision"] = config.model.revision
    result["example_pool_hash"] = example_pool_hash(
        item for item in examples if item.split == "train"
    )
    result["seed"] = config.seed
    result["elapsed_seconds"] = time.perf_counter() - run_started
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
