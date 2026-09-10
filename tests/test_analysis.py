from __future__ import annotations

from fedllm_heterogeneity.analysis import analyze_runs


def test_analysis_normalizes_and_recovers_factorial_effects():
    runs = []
    for seed in (1, 2, 3):
        runs.extend(
            [
                {"seed": seed, "arm": "base", "validation_nll": {"d/t": 2.0}},
                {"seed": seed, "arm": "centralized", "validation_nll": {"d/t": 1.0}},
            ]
        )
        recovery = {
            "iid": 1.00,
            "domain_only": 0.98,
            "task_only": 0.90,
            "coupled": 0.85,
        }
        for regime, value in recovery.items():
            runs.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "aggregation": "fedex_lora",
                    "validation_nll": {"d/t": 2.0 - value},
                    "_path": f"{seed}/{regime}",
                }
            )
    report = analyze_runs(runs, equivalence_margin=-0.05)
    effects = report["factorial_effects"]["fedex_lora"]
    assert effects["domain_skew_effect"]["mean"] < 0
    assert effects["task_skew_effect"]["mean"] < effects["domain_skew_effect"]["mean"]
    assert report["equivalence"]["fedex_lora"]["equivalent"]

