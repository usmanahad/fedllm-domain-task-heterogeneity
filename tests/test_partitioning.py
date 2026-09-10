from __future__ import annotations

from fedllm_heterogeneity.data import BuildCounts, build_controlled_examples
from fedllm_heterogeneity.partitioning import audit_partitions, build_partitions

from test_data import WhitespaceTokenizer, sources_by_domain


def test_all_regimes_cover_the_same_pool_exactly_once():
    examples = build_controlled_examples(
        sources_by_domain(size=8),
        WhitespaceTokenizer(),
        BuildCounts(train=4, validation=1, test=1),
        seed=42,
    )
    hashes = set()
    for regime in ("iid", "domain_only", "task_only", "coupled"):
        spec = build_partitions(examples, regime, num_clients=16, seed=42)
        report = audit_partitions(examples, spec)
        hashes.add(spec.example_pool_hash)
        assert report["unknown_ids"] == []
        assert report["missing_ids"] == []
        assert report["non_unique_assignments"] == []
        assert report["size_range"][1] - report["size_range"][0] <= 1
    assert len(hashes) == 1


def test_coupled_clients_contain_one_cell():
    examples = build_controlled_examples(
        sources_by_domain(size=8),
        WhitespaceTokenizer(),
        BuildCounts(train=4, validation=1, test=1),
        seed=3,
    )
    spec = build_partitions(examples, "coupled", num_clients=16, seed=3)
    assert all(len(weights) == 1 for weights in spec.cell_weights.values())

