#!/usr/bin/env python3
"""Create a small, results-only archive for download from Kaggle."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


RESULT_FILES = {"summary.json", "rounds.json", "analysis.json"}
TABULAR_SUFFIXES = {".csv", ".tsv"}
PROVENANCE_FILES = (
    "pyproject.toml",
    "requirements-kaggle-p100.txt",
    "requirements-kaggle-p100-flower.txt",
)


def is_result(path: Path, *, include_smoke: bool) -> bool:
    """Return whether an artifact is useful without model/data reconstruction."""

    lowered_parts = tuple(part.lower() for part in path.parts)
    if not include_smoke and any("smoke" in part for part in lowered_parts):
        return False
    if path.name in RESULT_FILES or path.name.endswith(".manifest.json"):
        return True
    if path.suffix.lower() in TABULAR_SUFFIXES:
        return True
    return path.suffix.lower() == ".json" and any(
        "eval" in part for part in lowered_parts
    )


def readable_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def git_revision(project: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def package_results(
    project: Path,
    artifacts: Path,
    output: Path,
    *,
    include_smoke: bool = False,
) -> tuple[int, int, int]:
    project = project.resolve()
    artifacts = artifacts.resolve()
    output = output.resolve()
    if not artifacts.is_dir():
        raise FileNotFoundError(f"Artifacts directory does not exist: {artifacts}")
    output.parent.mkdir(parents=True, exist_ok=True)

    candidates = sorted(path for path in artifacts.rglob("*") if path.is_file())
    selected = [
        path for path in candidates if path != output and is_result(path.relative_to(artifacts), include_smoke=include_smoke)
    ]
    if not selected:
        raise RuntimeError("No JSON summaries, diagnostics, manifests, or tabular results found")

    included_bytes = sum(path.stat().st_size for path in selected)
    omitted_bytes = sum(path.stat().st_size for path in candidates if path not in selected)
    project_artifacts = project / "artifacts"

    def result_archive_path(path: Path) -> Path:
        try:
            return Path("artifacts") / path.relative_to(project_artifacts)
        except ValueError:
            return Path("artifacts") / path.relative_to(artifacts)

    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        written: set[str] = set()

        def add_file(path: Path, archive_path: Path) -> None:
            key = archive_path.as_posix()
            if key not in written:
                archive.write(path, archive_path)
                written.add(key)

        for path in selected:
            add_file(path, result_archive_path(path))
        data_manifest = project / "artifacts/controlled_examples.jsonl.manifest.json"
        if data_manifest.is_file():
            add_file(data_manifest, Path("provenance/data-manifest.json"))
        seed_root = next(
            (
                path
                for path in (artifacts, *artifacts.parents)
                if path.name.startswith("seed-")
            ),
            None,
        )
        if seed_root is not None:
            for arm in ("base", "centralized"):
                reference = seed_root / arm / "summary.json"
                if reference.is_file():
                    add_file(reference, Path("references") / arm / "summary.json")
        for relative in PROVENANCE_FILES:
            path = project / relative
            if path.is_file():
                add_file(path, Path("provenance") / relative)
        configs = project / "configs"
        if configs.is_dir():
            for path in sorted(configs.glob("*.yaml")):
                add_file(path, Path("provenance/configs") / path.name)
        archive.writestr("provenance/git-commit.txt", git_revision(project) + "\n")

    return len(selected), included_bytes, omitted_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/fedllm-results.zip"),
    )
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    artifacts = args.artifacts
    if not artifacts.is_absolute():
        artifacts = args.project / artifacts
    count, included, omitted = package_results(
        args.project,
        artifacts,
        args.output,
        include_smoke=args.include_smoke,
    )
    print(f"Archive: {args.output.resolve()}")
    print(f"Result files: {count}")
    print(f"Uncompressed result data: {readable_size(included)}")
    print(f"Excluded adapters/data/smoke artifacts: {readable_size(omitted)}")
    print(f"Compressed archive: {readable_size(args.output.resolve().stat().st_size)}")


if __name__ == "__main__":
    main()
