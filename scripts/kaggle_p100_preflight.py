#!/usr/bin/env python3
"""Fail fast unless the Kaggle runtime can really execute CUDA on a P100."""

from __future__ import annotations

import importlib.metadata
import platform
import sys

import torch


PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "peft",
    "transformers",
)


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    for package in PACKAGES:
        print(f"{package}: {version(package)}")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. In Kaggle select Settings > Accelerator > GPU P100, "
            "then restart the session."
        )

    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    architectures = torch.cuda.get_arch_list()
    total_gib = torch.cuda.get_device_properties(index).total_memory / 2**30
    print(f"GPU: {name}")
    print(f"Compute capability: {capability[0]}.{capability[1]}")
    print(f"VRAM: {total_gib:.2f} GiB")
    print(f"Wheel architectures: {architectures}")

    required_arch = f"sm_{capability[0]}{capability[1]}"
    if required_arch not in architectures:
        raise SystemExit(
            f"This PyTorch wheel has no {required_arch} kernels. Run the CUDA 12.1 "
            "PyTorch 2.5.1 install cell in KAGGLE_P100.md and restart the kernel."
        )
    if capability != (6, 0):
        print("WARNING: this is not a Tesla P100; the experiment can still run.")

    # torch.cuda.is_available() alone is insufficient: execute an actual Pascal kernel.
    left = torch.randn((1024, 1024), device="cuda", dtype=torch.float16, requires_grad=True)
    right = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    loss = (left @ right).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    del left, right, loss
    torch.cuda.empty_cache()
    print("CUDA fp16 forward/backward: PASS")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()} (the configs use fp16)")
    print("P100 preflight: PASS")


if __name__ == "__main__":
    main()

