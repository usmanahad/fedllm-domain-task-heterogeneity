#!/usr/bin/env python3
"""Create a compact source-row split anchor from a canonical JSONL pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    source_assignments: dict[tuple[str, int], set[str]] = defaultdict(set)
    with args.examples.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            example = json.loads(line)
            domain = str(example["domain"])
            split = str(example["split"])
            row_index = int(example["metadata"]["row_index"])
            rows[domain][split].add(row_index)
            source_assignments[domain, row_index].add(split)
    overlap = {
        f"{domain}/{row_index}": sorted(splits)
        for (domain, row_index), splits in source_assignments.items()
        if len(splits) != 1
    }
    if overlap:
        raise ValueError(f"Sources occur in multiple splits: {overlap}")

    payload = {
        "schema_version": 1,
        "source_examples_sha256": sha256_file(args.examples),
        "identity": "Pinned Hugging Face dataset row index within each domain",
        "splits": {
            domain: {
                split: sorted(indices)
                for split, indices in sorted(domain_splits.items())
            }
            for domain, domain_splits in sorted(rows.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {args.output} with "
        f"{sum(len(v) for domain in rows.values() for v in domain.values())} source rows."
    )


if __name__ == "__main__":
    main()
