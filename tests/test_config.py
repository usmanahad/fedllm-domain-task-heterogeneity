from __future__ import annotations

from pathlib import Path

from fedllm_heterogeneity.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_existing_config_defaults_to_legacy_data_version():
    config = load_config(ROOT / "configs" / "controlled_qwen_p100.yaml")
    assert config.data_version == "legacy_v1"
    assert config.split_anchor is None


def test_clean_pilot_is_pinned_and_has_no_expensive_diagnostics():
    config = load_config(ROOT / "configs" / "controlled_qwen_p100_clean_v2.yaml")
    assert config.data_version == "clean_v2"
    assert config.split_anchor == "controlled_source_splits_v1.json"
    assert all(config.revisions.values())
    assert config.model.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert config.regimes == ("iid", "domain_only")
    assert config.aggregators == ("factor_fedavg", "fedex_lora")
    assert config.diagnostic_rounds == ()
    assert config.leave_one_out_rounds == ()
