"""Command-line entry points for data preparation, audits, and experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from .config import load_config
from .analysis import analyze_runs, load_run_summaries
from .data import (
    audit_examples,
    audit_known_native_boilerplate,
    audit_split_anchor,
    build_controlled_examples,
    build_native_examples,
    load_flowertune_sources,
    read_examples,
    write_examples,
)
from .lora import effective_lora_delta, factor_fedavg, fedex_lora, svd_effective_aggregate, weighted_effective_average
from .diagnostics import shared_token_mass, token_distribution, token_js_divergence
from .native_eval import export_merged_model, native_eval_command, run_native_evaluation
from .partitioning import audit_partitions, build_partitions, read_partitions, write_partitions
from .runner import run_federated, run_reference


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def command_build_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with pip install -e '.[train]'") from exc
    tokenizer_path = snapshot_download(
        config.model.model_id,
        revision=config.model.revision,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, fix_mistral_regex=False
    )
    sources = {
        domain: list(
            load_flowertune_sources(
                domain,
                dataset_name=dataset,
                revision=config.revisions.get(domain),
                cache_dir=args.cache_dir,
                data_version=config.data_version if config.track == "controlled" else "legacy_v1",
            )
        )
        for domain, dataset in config.datasets.items()
    }
    split_anchor = None
    split_anchor_path = None
    if config.track == "controlled" and config.split_anchor:
        split_anchor_path = Path(config.split_anchor)
        if not split_anchor_path.is_absolute():
            direct = split_anchor_path.resolve()
            relative_to_config = Path(args.config).resolve().parent / split_anchor_path
            split_anchor_path = direct if direct.exists() else relative_to_config
        split_anchor = json.loads(split_anchor_path.read_text())["splits"]
    if config.track == "controlled":
        examples = build_controlled_examples(
            sources,
            tokenizer,
            config.counts,
            config.seed,
            split_anchor=split_anchor,
        )
    else:
        examples = build_native_examples(sources, config.counts, config.seed)
    examples = [example for example in examples if example.task in config.tasks]
    digest = write_examples(args.output, examples)
    train_tokens = {}
    fertility = {}
    for domain in sorted(config.datasets):
        domain_examples = [
            item for item in examples if item.domain == domain and item.split == "train"
        ]
        sequences = [
            tokenizer.encode(item.prompt + "\n" + item.target, add_special_tokens=False)
            for item in domain_examples
        ]
        train_tokens[domain] = token_distribution(sequences)
        words = sum(len((item.prompt + " " + item.target).split()) for item in domain_examples)
        fertility[domain] = sum(len(sequence) for sequence in sequences) / max(words, 1)
    pairwise_tokens = {}
    domains = sorted(train_tokens)
    for index, left in enumerate(domains):
        for right in domains[index + 1 :]:
            pairwise_tokens[f"{left}:{right}"] = {
                "js_divergence": token_js_divergence(train_tokens[left], train_tokens[right]),
                "shared_token_mass": shared_token_mass(train_tokens[left], train_tokens[right]),
            }
    manifest = {
        "config": str(Path(args.config).resolve()),
        "track": config.track,
        "data_version": config.data_version,
        "seed": config.seed,
        "model_tokenizer": config.model.model_id,
        "model_revision": config.model.revision,
        "datasets": dict(config.datasets),
        "revisions": dict(config.revisions),
        "split_anchor": (
            {
                "path": str(split_anchor_path),
                "sha256": hashlib.sha256(split_anchor_path.read_bytes()).hexdigest(),
            }
            if split_anchor_path is not None
            else None
        ),
        "counts": asdict(config.counts),
        "num_examples": len(examples),
        "sha256": digest,
        "audit": audit_examples(examples),
        "known_native_boilerplate_audit": audit_known_native_boilerplate(examples),
        "split_anchor_retention": (
            audit_split_anchor(examples, split_anchor)
            if split_anchor is not None
            else None
        ),
        "token_statistics": {
            "fertility_tokens_per_whitespace_word": fertility,
            "pairwise": pairwise_tokens,
        },
    }
    manifest_path = Path(args.output).with_suffix(Path(args.output).suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _json(manifest)


def command_build_partitions(args: argparse.Namespace) -> None:
    examples = read_examples(args.examples)
    spec = build_partitions(examples, args.regime, args.clients, args.seed, args.split)
    write_partitions(args.output, spec)
    _json(audit_partitions(examples, spec))


def command_audit(args: argparse.Namespace) -> None:
    if args.expected_examples_sha256 is not None:
        actual_digest = hashlib.sha256(Path(args.examples).read_bytes()).hexdigest()
        if actual_digest != args.expected_examples_sha256:
            raise ValueError(
                f"Example JSONL SHA-256 is {actual_digest}; "
                f"expected {args.expected_examples_sha256}"
            )
    examples = read_examples(args.examples)
    report: dict[str, object] = {"examples": audit_examples(examples)}
    if args.partitions:
        spec = read_partitions(args.partitions)
        partition_report = audit_partitions(examples, spec)
        report["partitions"] = partition_report
        if args.expected_regime is not None and spec.regime != args.expected_regime:
            raise ValueError(
                f"Partition regime is {spec.regime}; expected {args.expected_regime}"
            )
        if (
            args.expected_partition_seed is not None
            and spec.seed != args.expected_partition_seed
        ):
            raise ValueError(
                f"Partition seed is {spec.seed}; expected {args.expected_partition_seed}"
            )
        if args.strict and (
            not partition_report["pool_hash_matches"]
            or partition_report["unknown_ids"]
            or partition_report["missing_ids"]
            or partition_report["non_unique_assignments"]
        ):
            raise ValueError("Partition integrity audit failed")
    _json(report)


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))


def command_lora_demo(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    states = []
    for _ in range(args.clients):
        states.append(
            {
                "layer.lora_A.default.weight": torch.randn(args.rank, args.input_dim),
                "layer.lora_B.default.weight": torch.randn(args.output_dim, args.rank),
            }
        )
    weights = [1.0] * len(states)
    target = weighted_effective_average(states, weights)["layer"]
    factor = effective_lora_delta(factor_fedavg(states, weights))["layer"]
    fedex = fedex_lora(states, weights)
    svd = svd_effective_aggregate(states, weights, rank=args.rank)
    _json(
        {
            "factor_fedavg_relative_error": _relative(factor, target),
            "fedex_relative_error": _relative(fedex.effective_delta["layer"], target),
            "svd_relative_error": _relative(svd.effective_delta["layer"], target),
            "rank": args.rank,
            "clients": args.clients,
        }
    )


def command_run_local(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.seed is not None:
        from dataclasses import replace

        config = replace(config, seed=args.seed, train=replace(config.train, seed=args.seed))
    examples = read_examples(args.examples)
    partitions = read_partitions(args.partitions)
    summary = run_federated(
        config,
        examples,
        partitions,
        args.aggregation,
        args.output,
        device=args.device,
    )
    _json(summary)


def command_run_reference(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.seed is not None:
        from dataclasses import replace

        config = replace(config, seed=args.seed, train=replace(config.train, seed=args.seed))
    examples = read_examples(args.examples)
    partitions = read_partitions(args.partitions) if args.partitions else None
    summary = run_reference(
        config,
        examples,
        args.arm,
        args.output,
        partitions=partitions,
        device=args.device,
    )
    _json(summary)


def command_analyze(args: argparse.Namespace) -> None:
    report = analyze_runs(
        load_run_summaries(args.runs),
        equivalence_margin=args.equivalence_margin,
        seed=args.seed,
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _json(report)


def command_export_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output = export_merged_model(
        config, args.adapter, args.base_residual, args.output
    )
    _json({"merged_model": str(output.resolve())})


def command_native_eval(args: argparse.Namespace) -> None:
    command = native_eval_command(
        args.model,
        args.suite,
        args.domain,
        args.output,
        batch_size=args.batch_size,
        allow_code_execution=args.allow_code_execution,
    )
    if args.dry_run:
        _json({"command": command})
        return
    run_native_evaluation(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fedllm-heterogeneity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_data = subparsers.add_parser("build-data")
    build_data.add_argument("--config", required=True)
    build_data.add_argument("--output", required=True)
    build_data.add_argument("--cache-dir")
    build_data.set_defaults(func=command_build_data)

    partitions = subparsers.add_parser("build-partitions")
    partitions.add_argument("--examples", required=True)
    partitions.add_argument("--regime", required=True, choices=["iid", "domain_only", "task_only", "coupled"])
    partitions.add_argument("--clients", type=int, default=16)
    partitions.add_argument("--seed", type=int, default=42)
    partitions.add_argument("--split", default="train")
    partitions.add_argument("--output", required=True)
    partitions.set_defaults(func=command_build_partitions)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--examples", required=True)
    audit.add_argument("--partitions")
    audit.add_argument("--expected-regime", choices=["iid", "domain_only", "task_only", "coupled"])
    audit.add_argument("--expected-partition-seed", type=int)
    audit.add_argument("--expected-examples-sha256")
    audit.add_argument("--strict", action="store_true")
    audit.set_defaults(func=command_audit)

    demo = subparsers.add_parser("lora-demo")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--clients", type=int, default=4)
    demo.add_argument("--rank", type=int, default=4)
    demo.add_argument("--input-dim", type=int, default=12)
    demo.add_argument("--output-dim", type=int, default=10)
    demo.set_defaults(func=command_lora_demo)

    run_local = subparsers.add_parser("run-local")
    run_local.add_argument("--config", required=True)
    run_local.add_argument("--examples", required=True)
    run_local.add_argument("--partitions", required=True)
    run_local.add_argument("--aggregation", default="fedex_lora")
    run_local.add_argument("--output", required=True)
    run_local.add_argument("--device")
    run_local.add_argument("--seed", type=int)
    run_local.set_defaults(func=command_run_local)

    reference = subparsers.add_parser("run-reference")
    reference.add_argument("--config", required=True)
    reference.add_argument("--examples", required=True)
    reference.add_argument("--arm", required=True, choices=["base", "centralized", "local_only"])
    reference.add_argument("--partitions")
    reference.add_argument("--output", required=True)
    reference.add_argument("--device")
    reference.add_argument("--seed", type=int)
    reference.set_defaults(func=command_run_reference)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--runs", required=True)
    analyze.add_argument("--equivalence-margin", type=float, default=-0.05)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--output")
    analyze.set_defaults(func=command_analyze)

    export = subparsers.add_parser("export-model")
    export.add_argument("--config", required=True)
    export.add_argument("--adapter", required=True)
    export.add_argument("--base-residual", required=True)
    export.add_argument("--output", required=True)
    export.set_defaults(func=command_export_model)

    native = subparsers.add_parser("native-eval")
    native.add_argument("--model", required=True)
    native.add_argument("--suite", default="configs/native_evaluation.yaml")
    native.add_argument("--domain", required=True, choices=["general", "finance", "medical", "code"])
    native.add_argument("--output", required=True)
    native.add_argument("--batch-size", default="auto")
    native.add_argument("--allow-code-execution", action="store_true")
    native.add_argument("--dry-run", action="store_true")
    native.set_defaults(func=command_native_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
