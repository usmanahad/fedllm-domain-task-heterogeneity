"""Native task metrics and cross-cell aggregation helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from .types import CanonicalExample


FLOWERTUNE_NATIVE_TASKS: dict[str, tuple[str, ...]] = {
    "general": ("mmlu",),
    "finance": ("financial_phrasebank", "fiqa_sa", "tfns"),
    "medical": ("pubmedqa", "medmcqa", "medqa_4options", "careqa"),
    "code": ("mbpp", "humaneval", "multiple_py"),
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def exact_match(prediction: str, target: str) -> float:
    return float(normalize_text(prediction) == normalize_text(target))


def token_f1(prediction: str, target: str) -> float:
    predicted = normalize_text(prediction).split()
    expected = normalize_text(target).split()
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for token in expected:
        counts[token] += 1
    overlap = 0
    for token in predicted:
        if counts[token] > 0:
            overlap += 1
            counts[token] -= 1
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def per_cell_metrics(
    examples: Sequence[CanonicalExample], predictions: Mapping[str, str]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for example in examples:
        if example.example_id in predictions:
            grouped[f"{example.domain}/{example.task}"].append(
                (predictions[example.example_id], example.target)
            )
    return {
        cell: {
            "exact_match": float(np.mean([exact_match(p, t) for p, t in values])),
            "token_f1": float(np.mean([token_f1(p, t) for p, t in values])),
            "num_examples": float(len(values)),
        }
        for cell, values in sorted(grouped.items())
    }


def normalized_recovery(
    score: float,
    base: float,
    centralized: float,
    higher_is_better: bool = True,
    minimum_denominator: float = 1e-8,
) -> float | None:
    numerator = score - base if higher_is_better else base - score
    denominator = centralized - base if higher_is_better else base - centralized
    if denominator <= minimum_denominator:
        return None
    return float(numerator / denominator)


def macro_average(values: Iterable[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(usable)) if usable else None

