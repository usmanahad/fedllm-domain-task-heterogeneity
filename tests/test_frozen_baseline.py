from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "fedllm-results"
FROZEN = ROOT / "baseline" / "fedex-qwen-3seed.json"


def _module():
    path = ROOT / "scripts" / "freeze_baseline.py"
    spec = importlib.util.spec_from_file_location("freeze_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not RESULTS.exists(), reason="downloaded baseline results are not present")
def test_downloaded_results_match_frozen_headlines_and_hashes():
    _module().verify(RESULTS, FROZEN)
