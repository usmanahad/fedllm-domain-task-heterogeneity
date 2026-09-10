from __future__ import annotations

import pytest

from fedllm_heterogeneity.native_eval import native_eval_command


def test_native_eval_command_and_code_safety(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "general:\n  tasks: [mmlu]\n"
        "code:\n  tasks: [humaneval]\n"
    )
    command = native_eval_command(tmp_path, suite, "general", tmp_path / "out")
    assert "mmlu" in command
    with pytest.raises(ValueError):
        native_eval_command(tmp_path, suite, "code", tmp_path / "out")
    unsafe = native_eval_command(
        tmp_path, suite, "code", tmp_path / "out", allow_code_execution=True
    )
    assert "--confirm_run_unsafe_code" in unsafe

