"""Stable experiment records shared by data, training, and analysis code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class Domain(str, Enum):
    GENERAL = "general"
    FINANCE = "finance"
    MEDICAL = "medical"
    CODE = "code"


class Task(str, Enum):
    CONTINUATION = "continuation"
    SPAN_RECONSTRUCTION = "span_reconstruction"
    NATIVE = "native"


class PartitionRegime(str, Enum):
    IID = "iid"
    DOMAIN_ONLY = "domain_only"
    TASK_ONLY = "task_only"
    COUPLED = "coupled"


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class CanonicalExample:
    """One model-ready example with enough provenance to reproduce it."""

    example_id: str
    source_id: str
    domain: str
    task: str
    prompt: str
    target: str
    split: str
    seed: int
    source_dataset: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalExample":
        return cls(**dict(value))


@dataclass(frozen=True)
class PartitionSpec:
    """Assignment of example IDs to clients for one partition regime."""

    regime: str
    seed: int
    num_clients: int
    assignments: Mapping[str, tuple[str, ...]]
    cell_weights: Mapping[str, Mapping[str, float]]
    example_pool_hash: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["assignments"] = {
            key: list(value) for key, value in self.assignments.items()
        }
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PartitionSpec":
        copied = dict(value)
        copied["assignments"] = {
            str(key): tuple(items)
            for key, items in dict(copied["assignments"]).items()
        }
        return cls(**copied)


@dataclass
class ClientUpdate:
    """A client update plus the accounting needed for fair aggregation."""

    round_id: int
    client_id: str
    num_examples: int
    num_input_tokens: int
    num_target_tokens: int
    adapter_state: Mapping[str, Any]
    effective_delta: Mapping[str, Any]
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregationDiagnostics:
    method: str
    client_weights: tuple[float, ...]
    factor_average_error: Mapping[str, float] = field(default_factory=dict)
    reconstruction_error: Mapping[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AggregationResult:
    adapter_state: Mapping[str, Any]
    base_residual: Mapping[str, Any]
    effective_delta: Mapping[str, Any]
    diagnostics: AggregationDiagnostics


@dataclass(frozen=True)
class SymmetryProfile:
    """Deterministic token-symmetry summary for one data split and text scope.

    ``groups`` are normally the eight ``domain/task`` cells.  Counts and
    distributions are kept as JSON-friendly summaries so the profile can be
    archived without retaining examples or model checkpoints.
    """

    split: str
    scope: str
    tokenizer: str
    groups: tuple[str, ...]
    example_counts: Mapping[str, int]
    token_counts: Mapping[str, int]
    vocabulary_sizes: Mapping[str, int]
    fertility_tokens_per_whitespace_unit: Mapping[str, float]
    pairwise: tuple[Mapping[str, Any], ...]
    domain_information: Mapping[str, Any] = field(default_factory=dict)
