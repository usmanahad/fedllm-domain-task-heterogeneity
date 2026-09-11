#!/usr/bin/env python3
"""Print the pre-registered Kaggle run commands without executing them."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from fedllm_heterogeneity.config import load_config


def command(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlled_qwen.yaml")
    parser.add_argument("--examples", default="artifacts/controlled_examples.jsonl")
    parser.add_argument("--output-root", default="artifacts/runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--data-tag", default="legacy-v1")
    args = parser.parse_args()
    config = load_config(args.config)
    root = (
        Path(args.output_root)
        / args.data_tag
        / f"partition-seed-{args.partition_seed}"
    )

    for seed in args.seeds:
        print(
            command(
                [
                    "fedllm-heterogeneity",
                    "run-reference",
                    "--config",
                    args.config,
                    "--examples",
                    args.examples,
                    "--arm",
                    "base",
                    "--seed",
                    seed,
                    "--output",
                    root / f"seed-{seed}" / "base",
                ]
            )
        )
        print(
            command(
                [
                    "fedllm-heterogeneity",
                    "run-reference",
                    "--config",
                    args.config,
                    "--examples",
                    args.examples,
                    "--arm",
                    "centralized",
                    "--seed",
                    seed,
                    "--output",
                    root / f"seed-{seed}" / "centralized",
                ]
            )
        )
        for regime in config.regimes:
            partitions = (
                f"artifacts/partitions-{args.data_tag}-"
                f"pseed-{args.partition_seed}-{regime}.json"
            )
            if regime == "coupled":
                print(
                    command(
                        [
                            "fedllm-heterogeneity",
                            "run-reference",
                            "--config",
                            args.config,
                            "--examples",
                            args.examples,
                            "--partitions",
                            partitions,
                            "--arm",
                            "local_only",
                            "--seed",
                            seed,
                            "--output",
                            root / f"seed-{seed}" / "local-only",
                        ]
                    )
                )
            for aggregation in config.aggregators:
                print(
                    command(
                        [
                            "fedllm-heterogeneity",
                            "run-local",
                            "--config",
                            args.config,
                            "--examples",
                            args.examples,
                            "--partitions",
                            partitions,
                            "--aggregation",
                            aggregation,
                            "--seed",
                            seed,
                            "--output",
                            root / f"seed-{seed}" / regime / aggregation,
                        ]
                    )
                )


if __name__ == "__main__":
    main()
