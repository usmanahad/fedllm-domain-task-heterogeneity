from __future__ import annotations

import hashlib

import pytest

from fedllm_heterogeneity.cli import build_parser


def test_audit_rejects_unexpected_example_file_hash(tmp_path):
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        '{"example_id":"e","source_id":"s","domain":"general",'
        '"task":"continuation","prompt":"p","target":"t",'
        '"split":"train","seed":42,"source_dataset":"d"}\n'
    )
    parser = build_parser()
    good_digest = hashlib.sha256(examples.read_bytes()).hexdigest()
    valid = parser.parse_args(
        [
            "audit",
            "--examples",
            str(examples),
            "--expected-examples-sha256",
            good_digest,
            "--strict",
        ]
    )
    valid.func(valid)

    invalid = parser.parse_args(
        [
            "audit",
            "--examples",
            str(examples),
            "--expected-examples-sha256",
            "0" * 64,
            "--strict",
        ]
    )
    with pytest.raises(ValueError, match="Example JSONL SHA-256"):
        invalid.func(invalid)
