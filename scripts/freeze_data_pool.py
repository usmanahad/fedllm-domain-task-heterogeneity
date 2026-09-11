#!/usr/bin/env python3
"""Freeze a portable integrity record for a canonical data pool."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    actual_sha = sha256_file(args.examples)
    if actual_sha != source["sha256"]:
        raise ValueError("Example JSONL does not match its build manifest")
    audit = source["audit"]
    if audit["duplicate_example_ids"] or any(
        audit["source_overlap_across_splits"].values()
    ):
        raise ValueError("Example integrity audit failed")
    instruction_leaks = {
        key: value
        for key, value in source["known_native_boilerplate_audit"].items()
        if key.endswith("native_instruction_fragment")
    }
    if instruction_leaks:
        raise ValueError(f"Known native instruction fragments remain: {instruction_leaks}")
    training_ids = []
    with args.examples.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                example = json.loads(line)
                if example["split"] == "train":
                    training_ids.append(example["example_id"])
    training_pool_hash = hashlib.sha256(
        "\n".join(sorted(training_ids)).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "data_version": source["data_version"],
        "track": source["track"],
        "seed": source["seed"],
        "model_tokenizer": source["model_tokenizer"],
        "model_revision": source.get("model_revision"),
        "datasets": source["datasets"],
        "revisions": source["revisions"],
        "counts": source["counts"],
        "num_examples": source["num_examples"],
        "canonical_examples_sha256": actual_sha,
        "training_example_pool_hash": training_pool_hash,
        "split_anchor_sha256": source["split_anchor"]["sha256"],
        "split_anchor_retention": source["split_anchor_retention"],
        "known_native_boilerplate_audit": source["known_native_boilerplate_audit"],
        "audit": audit,
        "token_statistics": source["token_statistics"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Froze {source['data_version']} at {args.output}: {actual_sha}")


if __name__ == "__main__":
    main()
