"""YAML configuration loading with explicit, validated defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .data import BuildCounts, FLOWERTUNE_DATASETS
from .training import ModelConfig, TrainConfig


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    track: str
    seed: int
    num_clients: int
    rounds: int
    regimes: tuple[str, ...]
    aggregators: tuple[str, ...]
    datasets: Mapping[str, str]
    revisions: Mapping[str, str | None]
    tasks: tuple[str, ...]
    counts: BuildCounts
    model: ModelConfig
    train: TrainConfig
    equivalence_margin: float = -0.05
    worst_cell_margin: float = -0.10
    diagnostic_rounds: tuple[int, ...] = (1, 8)
    probe_examples_per_cell: int = 16
    gradient_probe_examples_per_client: int = 2
    leave_one_out_rounds: tuple[int, ...] = (8,)
    save_checkpoints: bool = True


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be a mapping")
    data = dict(raw.get("data", {}))
    model = dict(raw.get("model", {}))
    train = dict(raw.get("train", {}))
    analysis = dict(raw.get("analysis", {}))
    datasets = dict(FLOWERTUNE_DATASETS)
    datasets.update(data.get("datasets", {}))
    revisions = {domain: None for domain in datasets}
    revisions.update(data.get("revisions", {}))
    counts_raw = dict(data.get("counts", {}))
    counts = BuildCounts(
        train=int(counts_raw.get("train", 512)),
        validation=int(counts_raw.get("validation", 128)),
        test=int(counts_raw.get("test", 256)),
    )
    model_config = ModelConfig(
        model_id=str(model.get("id", "Qwen/Qwen2.5-0.5B-Instruct")),
        quantization_bits=model.get("quantization_bits"),
        lora_rank=int(model.get("lora_rank", 16)),
        lora_alpha=int(model.get("lora_alpha", 32)),
        lora_dropout=float(model.get("lora_dropout", 0.05)),
        lora_targets=tuple(model.get("lora_targets", ["q_proj", "v_proj"])),
        ffa_lora=bool(model.get("ffa_lora", False)),
        random_init=bool(model.get("random_init", False)),
    )
    train_config = TrainConfig(
        learning_rate=float(train.get("learning_rate", 2e-4)),
        batch_size=int(train.get("batch_size", 4)),
        eval_batch_size=int(train.get("eval_batch_size", train.get("batch_size", 4))),
        max_steps=int(train.get("local_steps", 10)),
        max_length=int(train.get("max_length", 512)),
        weight_decay=float(train.get("weight_decay", 0.0)),
        gradient_clip_norm=float(train.get("gradient_clip_norm", 1.0)),
        seed=int(raw.get("seed", 42)),
    )
    track = str(_require(raw, "track"))
    if track not in {"controlled", "natural"}:
        raise ValueError("track must be 'controlled' or 'natural'")
    return ExperimentConfig(
        name=str(raw.get("name", Path(path).stem)),
        track=track,
        seed=int(raw.get("seed", 42)),
        num_clients=int(raw.get("num_clients", 16)),
        rounds=int(raw.get("rounds", 8)),
        regimes=tuple(raw.get("regimes", ["iid", "domain_only", "task_only", "coupled"])),
        aggregators=tuple(raw.get("aggregators", ["fedex_lora"])),
        datasets=datasets,
        revisions=revisions,
        tasks=tuple(data.get("tasks", ["continuation", "span_reconstruction"] if track == "controlled" else ["native"])),
        counts=counts,
        model=model_config,
        train=train_config,
        equivalence_margin=float(analysis.get("equivalence_margin", -0.05)),
        worst_cell_margin=float(analysis.get("worst_cell_margin", -0.10)),
        diagnostic_rounds=tuple(int(value) for value in analysis.get("diagnostic_rounds", [1, int(raw.get("rounds", 8))])),
        probe_examples_per_cell=int(analysis.get("probe_examples_per_cell", 16)),
        gradient_probe_examples_per_client=int(
            analysis.get("gradient_probe_examples_per_client", train_config.batch_size)
        ),
        leave_one_out_rounds=tuple(
            int(value)
            for value in analysis.get("leave_one_out_rounds", [int(raw.get("rounds", 8))])
        ),
        save_checkpoints=bool(raw.get("save_checkpoints", True)),
    )
