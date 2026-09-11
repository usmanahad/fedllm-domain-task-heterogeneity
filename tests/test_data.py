from __future__ import annotations

from collections import defaultdict

from fedllm_heterogeneity.data import (
    BuildCounts,
    SourceRecord,
    adapt_flowertune_row,
    audit_examples,
    audit_known_native_boilerplate,
    build_controlled_examples,
    select_source_splits_with_anchor,
)
from fedllm_heterogeneity.types import CanonicalExample, Domain


class WhitespaceTokenizer:
    def __init__(self):
        self.tokens: dict[str, int] = {}
        self.reverse: dict[int, str] = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        result = []
        for token in text.split():
            if token not in self.tokens:
                index = len(self.tokens) + 1
                self.tokens[token] = index
                self.reverse[index] = token
            result.append(self.tokens[token])
        return result

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(self.reverse[index] for index in token_ids)


def sources_by_domain(size=6):
    result = defaultdict(list)
    for domain in Domain:
        for index in range(size):
            text = " ".join(f"{domain.value}_{index}_{position}" for position in range(64))
            result[domain.value].append(
                SourceRecord(
                    source_id=f"{domain.value}-{index}",
                    domain=domain.value,
                    dataset=f"dataset/{domain.value}",
                    text=text,
                    native_prompt=f"question {index}",
                    native_target=f"answer {index}",
                    row_index=index,
                )
            )
    return result


def test_finance_label_is_excluded_from_controlled_text():
    source = adapt_flowertune_row(
        "finance",
        {"instruction": "The company increased revenue substantially", "output": "positive"},
        0,
    )
    assert source is not None
    assert source.native_target == "positive"
    assert "positive" not in source.text


def test_clean_v2_removes_only_verified_native_boilerplate_and_preserves_id():
    finance_row = {
        "instruction": "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.",
        "input": "The company increased revenue substantially.",
        "output": "positive",
    }
    legacy = adapt_flowertune_row("finance", finance_row, 7, data_version="legacy_v1")
    clean = adapt_flowertune_row("finance", finance_row, 7, data_version="clean_v2")
    assert legacy is not None and clean is not None
    assert clean.source_id == legacy.source_id
    assert clean.text == finance_row["input"]
    assert clean.native_prompt == legacy.native_prompt
    assert clean.removed_native_boilerplate == "finance_sentiment_instruction"

    medical = adapt_flowertune_row(
        "medical",
        {
            "instruction": "Answer this question truthfully",
            "input": "What is X?",
            "output": "X is a test answer.",
        },
        9,
        data_version="clean_v2",
    )
    assert medical is not None
    assert medical.text == "What is X?\n\nX is a test answer."
    assert medical.removed_native_boilerplate == "medical_truthfulness_instruction"


def test_split_anchor_preserves_eligible_rows_and_fills_missing_rows():
    sources = [
        SourceRecord(str(index), "general", "dataset", "text", "q", "a", index)
        for index in (0, 2, 3, 4, 5)
    ]
    anchor = {"train": [0, 1], "validation": [2], "test": [3]}
    selected = select_source_splits_with_anchor(
        sources,
        BuildCounts(train=2, validation=1, test=1),
        seed=42,
        anchored_row_indices=anchor,
    )
    assert [source.row_index for source in selected["validation"]] == [2]
    assert [source.row_index for source in selected["test"]] == [3]
    assert 0 in [source.row_index for source in selected["train"]]
    all_rows = [source.row_index for values in selected.values() for source in values]
    assert len(all_rows) == len(set(all_rows)) == 4


def test_boilerplate_audit_checks_prompt_and_target():
    examples = [
        CanonicalExample(
            example_id="example",
            source_id="source",
            domain="finance",
            task="span_reconstruction",
            prompt="Please choose an answer from {negative/neutral/positive}.",
            target="answer from {negative/neutral",
            split="train",
            seed=42,
            source_dataset="dataset",
        )
    ]
    report = audit_known_native_boilerplate(examples)
    assert any("finance" in key and "prompt" in key for key in report)
    assert any("finance" in key and "target" in key for key in report)


def test_controlled_build_is_deterministic_and_split_safe():
    tokenizer = WhitespaceTokenizer()
    counts = BuildCounts(train=2, validation=1, test=1)
    first = build_controlled_examples(sources_by_domain(), tokenizer, counts, seed=9)
    second = build_controlled_examples(sources_by_domain(), tokenizer, counts, seed=9)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == 4 * counts.total * 2
    audit = audit_examples(first)
    assert audit["duplicate_example_ids"] == 0
    assert all(value == 0 for value in audit["source_overlap_across_splits"].values())
    assert set(item.task for item in first) == {"continuation", "span_reconstruction"}
    assert all(item.target for item in first)
