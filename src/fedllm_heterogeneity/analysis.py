"""Aggregate run artifacts into normalized recovery and factorial effects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .evaluation import normalized_recovery
from .statistics import paired_bootstrap_interval


def load_run_summaries(root: str | Path) -> list[dict[str, object]]:
    summaries = []
    for path in sorted(Path(root).glob("**/summary.json")):
        value = json.loads(path.read_text())
        if "validation_nll" not in value and "final_validation_nll" not in value:
            continue
        value["_path"] = str(path)
        value["validation_nll"] = value.get("validation_nll", value.get("final_validation_nll"))
        summaries.append(value)
    return summaries


def _mean_interval(values: Iterable[float], seed: int = 42) -> dict[str, float] | None:
    values = list(values)
    if not values:
        return None
    if len(values) == 1:
        return {"mean": values[0], "lower": values[0], "upper": values[0]}
    zeros = [0.0] * len(values)
    interval = paired_bootstrap_interval(values, zeros, seed=seed)
    return {"mean": interval.estimate, "lower": interval.lower, "upper": interval.upper}


def analyze_runs(
    summaries: list[dict[str, object]],
    equivalence_margin: float = -0.05,
    seed: int = 42,
) -> dict[str, object]:
    by_seed: dict[int, list[dict[str, object]]] = {}
    for summary in summaries:
        by_seed.setdefault(int(summary.get("seed", 0)), []).append(summary)

    normalized_runs: list[dict[str, object]] = []
    for run_seed, values in sorted(by_seed.items()):
        base = next((item for item in values if item.get("arm") == "base"), None)
        central = next((item for item in values if item.get("arm") == "centralized"), None)
        if base is None or central is None:
            continue
        base_nll = dict(base["validation_nll"])
        central_nll = dict(central["validation_nll"])
        for item in values:
            if "regime" not in item:
                continue
            cell_recovery = {
                cell: normalized_recovery(
                    float(value),
                    float(base_nll[cell]),
                    float(central_nll[cell]),
                    higher_is_better=False,
                )
                for cell, value in dict(item["validation_nll"]).items()
                if cell in base_nll and cell in central_nll
            }
            usable = [float(value) for value in cell_recovery.values() if value is not None]
            normalized_runs.append(
                {
                    "seed": run_seed,
                    "regime": item["regime"],
                    "aggregation": item["aggregation"],
                    "macro_recovery": float(np.mean(usable)) if usable else None,
                    "worst_cell_recovery": min(usable) if usable else None,
                    "cell_recovery": cell_recovery,
                    "source": item["_path"],
                }
            )

    factorial: dict[str, object] = {}
    equivalence: dict[str, object] = {}
    aggregators = sorted({str(item["aggregation"]) for item in normalized_runs})
    for aggregation in aggregators:
        selected = [item for item in normalized_runs if item["aggregation"] == aggregation]
        lookup = {
            (int(item["seed"]), str(item["regime"])): float(item["macro_recovery"])
            for item in selected
            if item["macro_recovery"] is not None
        }
        seeds = sorted({key[0] for key in lookup})
        domain_effects = []
        task_effects = []
        interactions = []
        iid_values = []
        domain_values = []
        for run_seed in seeds:
            required = [(run_seed, regime) for regime in ("iid", "domain_only", "task_only", "coupled")]
            if not all(key in lookup for key in required):
                continue
            iid = lookup[(run_seed, "iid")]
            domain = lookup[(run_seed, "domain_only")]
            task = lookup[(run_seed, "task_only")]
            coupled = lookup[(run_seed, "coupled")]
            iid_values.append(iid)
            domain_values.append(domain)
            domain_effects.append(domain - iid)
            task_effects.append(task - iid)
            interactions.append(coupled - domain - task + iid)
        factorial[aggregation] = {
            "domain_skew_effect": _mean_interval(domain_effects, seed),
            "task_skew_effect": _mean_interval(task_effects, seed),
            "domain_task_interaction": _mean_interval(interactions, seed),
            "num_complete_seeds": len(domain_effects),
        }
        if len(iid_values) >= 2:
            interval = paired_bootstrap_interval(domain_values, iid_values, seed=seed)
            equivalence[aggregation] = {
                "domain_only_minus_iid": {
                    "mean": interval.estimate,
                    "lower": interval.lower,
                    "upper": interval.upper,
                },
                "margin": equivalence_margin,
                "equivalent": interval.lower > equivalence_margin,
            }

    return {
        "num_summaries": len(summaries),
        "num_normalized_runs": len(normalized_runs),
        "normalized_runs": normalized_runs,
        "factorial_effects": factorial,
        "equivalence": equivalence,
        "notes": [
            "Recovery is normalized per cell against pretrained-base and centralized NLL.",
            "Factorial effects are paired seed-level contrasts; negative values indicate worse recovery.",
        ],
    }

