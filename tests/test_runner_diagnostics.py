from __future__ import annotations

import pytest

from fedllm_heterogeneity.runner import _leave_one_out_payload


def test_leave_one_out_payload_has_unambiguous_harm_sign():
    payload = _leave_one_out_payload(
        ["helpful", "harmful"],
        ["general/continuation", "medical/continuation"],
        {
            "general/continuation": 2.0,
            "medical/continuation": 3.0,
        },
        [
            [2.2, 3.2],  # Removing a helpful client makes loss worse.
            [1.9, 2.7],  # Removing a harmful client makes loss better.
        ],
    )

    assert payload["macro_loss_delta_without_client_minus_full"] == pytest.approx(
        [0.2, -0.2]
    )
    assert payload["client_harm_score"] == pytest.approx([-0.2, 0.2])


def test_leave_one_out_payload_rejects_incomplete_rows():
    with pytest.raises(ValueError, match="every evaluation cell"):
        _leave_one_out_payload(
            ["client-0"],
            ["cell-a", "cell-b"],
            {"cell-a": 1.0, "cell-b": 2.0},
            [[1.5]],
        )
