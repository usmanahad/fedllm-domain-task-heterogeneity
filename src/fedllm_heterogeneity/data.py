"""FlowerTune adapters and deterministic controlled-task construction."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Protocol, Sequence

from .types import CanonicalExample, Domain, Split, Task


FLOWERTUNE_DATASETS: dict[str, str] = {
    Domain.GENERAL.value: "flwrlabs/alpaca-gpt4",
    Domain.FINANCE.value: "flwrlabs/fingpt-sentiment-train",
    Domain.MEDICAL.value: "flwrlabs/medical-meadow-medical-flashcards",
    Domain.CODE.value: "flwrlabs/code-alpaca-20k",
}


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str: ...


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    domain: str
    dataset: str
    text: str
    native_prompt: str
    native_target: str
    row_index: int


@dataclass(frozen=True)
class BuildCounts:
    train: int = 512
    validation: int = 128
    test: int = 256

    @property
    def total(self) -> int:
        return self.train + self.validation + self.test


def stable_hash(*parts: object, length: int = 24) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _first(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = _clean(row.get(key))
        if value:
            return value
    return ""


def _join(*values: str) -> str:
    return "\n\n".join(value for value in values if value)


def adapt_flowertune_row(
    domain: str,
    row: Mapping[str, Any],
    row_index: int,
    dataset: str | None = None,
) -> SourceRecord | None:
    """Convert known FlowerTune schemas to a common, template-free record.

    The finance label is intentionally excluded from controlled ``text`` but is
    retained as the native target. Other domains retain their response because
    it contains domain text needed for open-vocabulary transformations.
    """

    domain = Domain(domain).value
    dataset = dataset or FLOWERTUNE_DATASETS[domain]
    instruction = _first(row, "instruction", "question", "prompt", "Question")
    extra_input = _first(row, "input", "context")
    response = _first(
        row, "response", "output", "answer", "completion", "Answer", "text"
    )

    if domain == Domain.FINANCE.value:
        source_text = _join(instruction, extra_input)
    else:
        source_text = _join(instruction, extra_input, response)

    native_prompt = _join(instruction, extra_input)
    if not source_text or not native_prompt or not response:
        return None

    source_id = stable_hash(dataset, row_index, source_text)
    return SourceRecord(
        source_id=source_id,
        domain=domain,
        dataset=dataset,
        text=source_text,
        native_prompt=native_prompt,
        native_target=response,
        row_index=row_index,
    )


def _rng(seed: int, source_id: str, task: str) -> random.Random:
    derived = int(stable_hash(seed, source_id, task, length=16), 16)
    return random.Random(derived)


def _decode(tokenizer: TokenizerLike, ids: Sequence[int]) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=False).strip()


def make_continuation(
    source: SourceRecord,
    tokenizer: TokenizerLike,
    split: str,
    seed: int,
    target_tokens: int = 8,
    min_context_tokens: int = 24,
    max_context_tokens: int = 96,
) -> CanonicalExample | None:
    ids = tokenizer.encode(source.text, add_special_tokens=False)
    if len(ids) < min_context_tokens + target_tokens:
        return None
    rng = _rng(seed, source.source_id, Task.CONTINUATION.value)
    context_len = rng.randint(
        min_context_tokens, min(max_context_tokens, len(ids) - target_tokens)
    )
    max_start = len(ids) - context_len - target_tokens
    start = rng.randint(0, max_start)
    context = _decode(tokenizer, ids[start : start + context_len])
    target = _decode(
        tokenizer,
        ids[start + context_len : start + context_len + target_tokens],
    )
    if not context or not target:
        return None
    task = Task.CONTINUATION.value
    return CanonicalExample(
        example_id=stable_hash(source.source_id, split, task, seed),
        source_id=source.source_id,
        domain=source.domain,
        task=task,
        prompt=f"Continue the following text.\n\n{context}\n\nContinuation:",
        target=target,
        split=Split(split).value,
        seed=seed,
        source_dataset=source.dataset,
        metadata={
            "row_index": source.row_index,
            "target_tokens": target_tokens,
            "context_tokens": context_len,
            "token_start": start,
        },
    )


def make_span_reconstruction(
    source: SourceRecord,
    tokenizer: TokenizerLike,
    split: str,
    seed: int,
    target_tokens: int = 8,
    context_side_tokens: int = 48,
) -> CanonicalExample | None:
    ids = tokenizer.encode(source.text, add_special_tokens=False)
    if len(ids) < (2 * target_tokens) + target_tokens:
        return None
    rng = _rng(seed, source.source_id, Task.SPAN_RECONSTRUCTION.value)
    span_start = rng.randint(target_tokens, len(ids) - (2 * target_tokens))
    span_end = span_start + target_tokens
    left_start = max(0, span_start - context_side_tokens)
    right_end = min(len(ids), span_end + context_side_tokens)
    left = _decode(tokenizer, ids[left_start:span_start])
    target = _decode(tokenizer, ids[span_start:span_end])
    right = _decode(tokenizer, ids[span_end:right_end])
    if not left or not right or not target:
        return None
    task = Task.SPAN_RECONSTRUCTION.value
    return CanonicalExample(
        example_id=stable_hash(source.source_id, split, task, seed),
        source_id=source.source_id,
        domain=source.domain,
        task=task,
        prompt=(
            "Reconstruct the missing text exactly.\n\n"
            f"{left} <MISSING> {right}\n\nMissing text:"
        ),
        target=target,
        split=Split(split).value,
        seed=seed,
        source_dataset=source.dataset,
        metadata={
            "row_index": source.row_index,
            "target_tokens": target_tokens,
            "left_context_tokens": span_start - left_start,
            "right_context_tokens": right_end - span_end,
            "token_start": span_start,
        },
    )


def make_native_example(
    source: SourceRecord, split: str, seed: int
) -> CanonicalExample:
    task = Task.NATIVE.value
    return CanonicalExample(
        example_id=stable_hash(source.source_id, split, task, seed),
        source_id=source.source_id,
        domain=source.domain,
        task=task,
        prompt=source.native_prompt,
        target=source.native_target,
        split=Split(split).value,
        seed=seed,
        source_dataset=source.dataset,
        metadata={"row_index": source.row_index},
    )


def select_source_splits(
    sources: Iterable[SourceRecord], counts: BuildCounts, seed: int
) -> dict[str, list[SourceRecord]]:
    """Select exact source counts in deterministic, disjoint splits."""

    unique: MutableMapping[str, SourceRecord] = {}
    for source in sources:
        unique.setdefault(source.source_id, source)
    ordered = sorted(
        unique.values(), key=lambda item: stable_hash(seed, item.source_id, "split")
    )
    if len(ordered) < counts.total:
        raise ValueError(
            f"Need {counts.total} eligible unique sources, found {len(ordered)}"
        )
    train_end = counts.train
    validation_end = train_end + counts.validation
    return {
        Split.TRAIN.value: ordered[:train_end],
        Split.VALIDATION.value: ordered[train_end:validation_end],
        Split.TEST.value: ordered[validation_end : validation_end + counts.test],
    }


def build_controlled_examples(
    sources_by_domain: Mapping[str, Iterable[SourceRecord]],
    tokenizer: TokenizerLike,
    counts: BuildCounts = BuildCounts(),
    seed: int = 42,
) -> list[CanonicalExample]:
    examples: list[CanonicalExample] = []
    for domain in Domain:
        sources = list(sources_by_domain[domain.value])
        eligible = [
            source
            for source in sources
            if len(tokenizer.encode(source.text, add_special_tokens=False)) >= 32
        ]
        splits = select_source_splits(eligible, counts, seed)
        for split, split_sources in splits.items():
            for source in split_sources:
                continuation = make_continuation(source, tokenizer, split, seed)
                reconstruction = make_span_reconstruction(
                    source, tokenizer, split, seed
                )
                if continuation is None or reconstruction is None:
                    raise ValueError(
                        "An eligible source failed transformation: "
                        f"{domain.value}/{source.source_id}"
                    )
                examples.extend((continuation, reconstruction))
    return examples


def build_native_examples(
    sources_by_domain: Mapping[str, Iterable[SourceRecord]],
    counts: BuildCounts = BuildCounts(),
    seed: int = 42,
) -> list[CanonicalExample]:
    """Build the natural FlowerTune instruction-response track."""

    examples: list[CanonicalExample] = []
    for domain in Domain:
        splits = select_source_splits(list(sources_by_domain[domain.value]), counts, seed)
        for split, sources in splits.items():
            examples.extend(make_native_example(source, split, seed) for source in sources)
    return examples


def load_flowertune_sources(
    domain: str,
    dataset_name: str | None = None,
    split: str = "train",
    revision: str | None = None,
    cache_dir: str | None = None,
) -> Iterator[SourceRecord]:
    """Download and adapt a FlowerTune dataset lazily.

    Importing ``datasets`` here keeps the research core dependency-light.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the training extras: pip install -e '.[train]'") from exc

    domain = Domain(domain).value
    dataset_name = dataset_name or FLOWERTUNE_DATASETS[domain]
    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    dataset = load_dataset(dataset_name, **kwargs)
    for index, row in enumerate(dataset):
        adapted = adapt_flowertune_row(domain, row, index, dataset_name)
        if adapted is not None:
            yield adapted


def write_examples(path: str | Path, examples: Iterable[CanonicalExample]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for example in sorted(examples, key=lambda item: item.example_id):
            line = json.dumps(example.to_dict(), sort_keys=True, ensure_ascii=False)
            handle.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def read_examples(path: str | Path) -> list[CanonicalExample]:
    with Path(path).open(encoding="utf-8") as handle:
        return [CanonicalExample.from_dict(json.loads(line)) for line in handle if line.strip()]


def audit_examples(examples: Sequence[CanonicalExample]) -> dict[str, Any]:
    ids = [example.example_id for example in examples]
    split_sources: dict[str, set[str]] = {}
    cell_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for example in examples:
        split_sources.setdefault(example.split, set()).add(example.source_id)
        cell_counts[f"{example.split}/{example.domain}/{example.task}"] += 1
        if example.task == Task.NATIVE.value:
            label_counts[example.target] += 1

    overlap: dict[str, int] = {}
    split_names = sorted(split_sources)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap[f"{left}:{right}"] = len(
                split_sources[left].intersection(split_sources[right])
            )

    return {
        "num_examples": len(examples),
        "num_unique_examples": len(set(ids)),
        "num_unique_sources": len({item.source_id for item in examples}),
        "duplicate_example_ids": len(ids) - len(set(ids)),
        "source_overlap_across_splits": overlap,
        "cell_counts": dict(sorted(cell_counts.items())),
        "native_target_counts": dict(label_counts.most_common(20)),
    }
