from __future__ import annotations

from collections import defaultdict

from fedllm_heterogeneity.data import (
    BuildCounts,
    SourceRecord,
    adapt_flowertune_row,
    audit_examples,
    build_controlled_examples,
)
from fedllm_heterogeneity.types import Domain


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

