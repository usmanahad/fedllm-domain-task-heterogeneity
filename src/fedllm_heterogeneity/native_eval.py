"""Export exact global models and invoke native FlowerTune evaluation suites."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import torch
import yaml

from .config import ExperimentConfig
from .lora import apply_base_residual
from .training import build_model_and_tokenizer, set_adapter_state


def export_merged_model(
    config: ExperimentConfig,
    adapter_path: str | Path,
    residual_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Materialize base residual + LoRA into a standard HF checkpoint."""

    if config.model.quantization_bits is not None:
        raise ValueError(
            "Export from a non-quantized model config; merging into a 4-bit base is intentionally disabled"
        )
    model, tokenizer = build_model_and_tokenizer(config.model, device="cpu")
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
    residual = torch.load(residual_path, map_location="cpu", weights_only=True)
    if residual:
        apply_base_residual(model, residual)
    set_adapter_state(model, adapter)
    if not hasattr(model, "merge_and_unload"):
        raise TypeError("Expected a PEFT model exposing merge_and_unload")
    merged = model.merge_and_unload()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "export_manifest.json").write_text(
        json.dumps(
            {
                "base_model": config.model.model_id,
                "adapter": str(Path(adapter_path).resolve()),
                "base_residual": str(Path(residual_path).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output


def native_eval_command(
    model_path: str | Path,
    suite_path: str | Path,
    domain: str,
    output_path: str | Path,
    batch_size: str = "auto",
    allow_code_execution: bool = False,
) -> list[str]:
    suites = yaml.safe_load(Path(suite_path).read_text())
    if domain not in suites:
        raise ValueError(f"Unknown evaluation domain: {domain}")
    tasks = list(suites[domain]["tasks"])
    command = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={Path(model_path).resolve()}",
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        batch_size,
        "--output_path",
        str(Path(output_path).resolve()),
    ]
    if domain == "code":
        if not allow_code_execution:
            raise ValueError(
                "Code benchmarks execute generated programs. Pass --allow-code-execution only in an isolated runtime."
            )
        command.append("--confirm_run_unsafe_code")
    return command


def run_native_evaluation(command: Sequence[str]) -> None:
    if shutil.which(command[0]) is None:
        raise RuntimeError("lm_eval is not installed; install the eval extra")
    subprocess.run(list(command), check=True)

