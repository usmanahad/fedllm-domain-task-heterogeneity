#!/usr/bin/env python3
"""Freeze or verify the compact 3-seed × 4-regime FedEx baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


RUN_PATTERN = re.compile(r"seed-(\d+)__(iid|domain_only|task_only|coupled)__fedex_lora$")
REGIMES = ("iid", "domain_only", "task_only", "coupled")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_manifest(results: Path) -> dict[str, object]:
    runs: dict[tuple[int, str], Path] = {}
    for path in sorted(results.iterdir()):
        match = RUN_PATTERN.fullmatch(path.name)
        if match:
            runs[int(match.group(1)), match.group(2)] = path
    seeds = sorted({seed for seed, _ in runs})
    expected = {(seed, regime) for seed in seeds for regime in REGIMES}
    if len(seeds) != 3 or set(runs) != expected:
        raise ValueError(f"Expected a complete 3 × 4 matrix; found {sorted(runs)}")

    files: list[dict[str, object]] = []
    recoveries: dict[tuple[int, str], float] = {}
    round8: dict[str, dict[str, list[float]]] = {
        regime: {
            "gradient_conflict_fraction": [],
            "update_conflict_fraction": [],
            "harmful_client_cell_fraction": [],
            "full_aggregate_loss_delta": [],
            "mean_individual_client_loss_delta": [],
        }
        for regime in REGIMES
    }
    pool_hashes: set[str] = set()
    data_hashes: set[str] = set()
    commits: set[str] = set()

    for seed in seeds:
        for regime in REGIMES:
            directory = runs[seed, regime]
            artifact = directory / "artifacts" / "runs" / f"seed-{seed}" / regime / "fedex_lora"
            summary_path = artifact / "summary.json"
            rounds_path = artifact / "rounds.json"
            for path in (summary_path, rounds_path):
                files.append(
                    {
                        "path": str(path.relative_to(results)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            summary = _load(summary_path)
            history = _load(rounds_path)
            if len(history) != 8 or history[-1]["round"] != 8:
                raise ValueError(f"Incomplete round history: {directory.name}")
            if len(history[-1]["leave_one_out_aggregation"]["client_ids"]) != 16:
                raise ValueError(f"Malformed LOO matrix: {directory.name}")

            base = _load(directory / "references" / "base" / "summary.json")["validation_nll"]
            centralized = _load(directory / "references" / "centralized" / "summary.json")["validation_nll"]
            final = summary["final_validation_nll"]
            cells = sorted(final)
            cell_recovery = [
                (base[cell] - final[cell]) / (base[cell] - centralized[cell]) for cell in cells
            ]
            recoveries[seed, regime] = float(np.mean(cell_recovery))

            diagnostic = history[-1]
            values = {
                "gradient_conflict_fraction": diagnostic["true_gradient_geometry"]["conflict_fraction"],
                "update_conflict_fraction": diagnostic["geometry"]["conflict_fraction"],
                "harmful_client_cell_fraction": diagnostic["functional_transfer"]["negative_transfer_fraction"],
                "full_aggregate_loss_delta": np.mean(list(diagnostic["functional_transfer"]["aggregate_loss_delta"].values())),
                "mean_individual_client_loss_delta": np.mean(list(diagnostic["functional_transfer"]["weighted_mean_client_loss_delta"].values())),
            }
            for name, value in values.items():
                round8[regime][name].append(float(value))
            pool_hashes.add(summary["example_pool_hash"])
            data_hashes.add(_load(directory / "provenance" / "data-manifest.json")["sha256"])
            commits.add((directory / "provenance" / "git-commit.txt").read_text().strip())

    if len(pool_hashes) != 1 or len(data_hashes) != 1 or len(commits) != 1:
        raise ValueError("Baseline does not share one pool hash, data hash, and Git commit")

    regime_means = {
        regime: float(np.mean([recoveries[seed, regime] for seed in seeds]))
        for regime in REGIMES
    }
    differences = {
        regime: float(np.mean([recoveries[seed, regime] - recoveries[seed, "iid"] for seed in seeds]))
        for regime in REGIMES[1:]
    }
    return {
        "schema_version": 1,
        "experiment": "Qwen2.5-0.5B-Instruct / FedEx-LoRA / 3 seeds / 4 regimes",
        "seeds": seeds,
        "regimes": list(REGIMES),
        "git_commit": next(iter(commits)),
        "canonical_examples_sha256": next(iter(data_hashes)),
        "training_example_pool_hash": next(iter(pool_hashes)),
        "files": files,
        "headline": {
            "macro_normalized_recovery": regime_means,
            "difference_vs_iid": differences,
            "round8_mean": {
                regime: {name: float(np.mean(values)) for name, values in metrics.items()}
                for regime, metrics in round8.items()
            },
        },
    }


def verify(results: Path, frozen_path: Path) -> None:
    frozen = _load(frozen_path)
    actual = build_manifest(results)
    if frozen != actual:
        raise AssertionError("Frozen baseline differs from current result files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("fedllm-results"))
    parser.add_argument("--output", type=Path, default=Path("baseline/fedex-qwen-3seed.json"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.results, args.output)
        print(f"Verified {args.output} against 12 result directories.")
        return
    manifest = build_manifest(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output} with {len(manifest['files'])} hashed result files.")


if __name__ == "__main__":
    main()
