"""Balanced client partitions over a fixed canonical example pool."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .data import stable_hash
from .types import CanonicalExample, Domain, PartitionRegime, PartitionSpec, Task


def example_pool_hash(examples: Iterable[CanonicalExample]) -> str:
    payload = "\n".join(sorted(example.example_id for example in examples))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _balanced_assign(
    examples: Sequence[CanonicalExample], client_ids: Sequence[str], seed: int, salt: str
) -> dict[str, list[str]]:
    result = {client_id: [] for client_id in client_ids}
    total_load: Counter[str] = Counter()
    cell_load: Counter[tuple[str, tuple[str, str]]] = Counter()
    by_cell: dict[tuple[str, str], list[CanonicalExample]] = defaultdict(list)
    for example in examples:
        by_cell[(example.domain, example.task)].append(example)
    for cell, members in sorted(by_cell.items()):
        ordered = sorted(
            members,
            key=lambda item: stable_hash(seed, salt, cell, item.example_id),
        )
        tie_order = sorted(
            client_ids,
            key=lambda client_id: stable_hash(seed, salt, cell, client_id),
        )
        tie_rank = {client_id: index for index, client_id in enumerate(tie_order)}
        for example in ordered:
            client_id = min(
                client_ids,
                key=lambda candidate: (
                    cell_load[(candidate, cell)],
                    total_load[candidate],
                    tie_rank[candidate],
                ),
            )
            result[client_id].append(example.example_id)
            cell_load[(client_id, cell)] += 1
            total_load[client_id] += 1
    return result


def _allowed_clients(regime: str, num_clients: int) -> Mapping[str, list[str]]:
    clients = [f"client_{index:02d}" for index in range(num_clients)]
    domains = [domain.value for domain in Domain]
    tasks = [Task.CONTINUATION.value, Task.SPAN_RECONSTRUCTION.value]
    if regime == PartitionRegime.IID.value:
        return {"*": clients}
    if regime == PartitionRegime.DOMAIN_ONLY.value:
        per_domain = num_clients // len(domains)
        return {
            domain: clients[index * per_domain : (index + 1) * per_domain]
            for index, domain in enumerate(domains)
        }
    if regime == PartitionRegime.TASK_ONLY.value:
        per_task = num_clients // len(tasks)
        return {
            task: clients[index * per_task : (index + 1) * per_task]
            for index, task in enumerate(tasks)
        }
    if regime == PartitionRegime.COUPLED.value:
        cells = [(domain, task) for domain in domains for task in tasks]
        per_cell = num_clients // len(cells)
        return {
            f"{domain}/{task}": clients[index * per_cell : (index + 1) * per_cell]
            for index, (domain, task) in enumerate(cells)
        }
    raise ValueError(f"Unknown partition regime: {regime}")


def build_partitions(
    examples: Sequence[CanonicalExample],
    regime: str,
    num_clients: int = 16,
    seed: int = 42,
    split: str = "train",
) -> PartitionSpec:
    """Build one of the pre-registered 16-client regimes."""

    regime = PartitionRegime(regime).value
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if regime == PartitionRegime.DOMAIN_ONLY.value and num_clients % 4:
        raise ValueError("domain_only requires a multiple of four clients")
    if regime == PartitionRegime.TASK_ONLY.value and num_clients % 2:
        raise ValueError("task_only requires a multiple of two clients")
    if regime == PartitionRegime.COUPLED.value and num_clients % 8:
        raise ValueError("coupled requires a multiple of eight clients")

    pool = [example for example in examples if example.split == split]
    allowed = _allowed_clients(regime, num_clients)
    assignments = {f"client_{index:02d}": [] for index in range(num_clients)}

    if regime == PartitionRegime.IID.value:
        assignments = _balanced_assign(pool, allowed["*"], seed, regime)
    else:
        grouped: dict[str, list[CanonicalExample]] = defaultdict(list)
        for example in pool:
            if regime == PartitionRegime.DOMAIN_ONLY.value:
                key = example.domain
            elif regime == PartitionRegime.TASK_ONLY.value:
                key = example.task
            else:
                key = f"{example.domain}/{example.task}"
            grouped[key].append(example)
        for key, members in grouped.items():
            local = _balanced_assign(members, allowed[key], seed, f"{regime}/{key}")
            for client_id, ids in local.items():
                assignments[client_id].extend(ids)

    lookup = {example.example_id: example for example in pool}
    cell_weights: dict[str, dict[str, float]] = {}
    for client_id, ids in assignments.items():
        counts = Counter(
            f"{lookup[example_id].domain}/{lookup[example_id].task}"
            for example_id in ids
        )
        total = sum(counts.values())
        cell_weights[client_id] = {
            cell: count / total for cell, count in sorted(counts.items())
        }

    return PartitionSpec(
        regime=regime,
        seed=seed,
        num_clients=num_clients,
        assignments={
            client_id: tuple(sorted(ids)) for client_id, ids in assignments.items()
        },
        cell_weights=cell_weights,
        example_pool_hash=example_pool_hash(pool),
    )


def write_partitions(path: str | Path, spec: PartitionSpec) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n")


def read_partitions(path: str | Path) -> PartitionSpec:
    return PartitionSpec.from_dict(json.loads(Path(path).read_text()))


def audit_partitions(
    examples: Sequence[CanonicalExample], spec: PartitionSpec
) -> dict[str, object]:
    training = {item.example_id: item for item in examples if item.split == "train"}
    assigned = [item for values in spec.assignments.values() for item in values]
    counts = Counter(assigned)
    unknown = sorted(set(assigned).difference(training))
    missing = sorted(set(training).difference(assigned))
    duplicates = sorted(item for item, count in counts.items() if count != 1)
    client_sizes = {key: len(value) for key, value in spec.assignments.items()}
    return {
        "regime": spec.regime,
        "pool_hash_matches": spec.example_pool_hash == example_pool_hash(training.values()),
        "unknown_ids": unknown,
        "missing_ids": missing,
        "non_unique_assignments": duplicates,
        "client_sizes": client_sizes,
        "size_range": [min(client_sizes.values()), max(client_sizes.values())],
        "cell_weights": spec.cell_weights,
    }
