import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def test_results_archive_excludes_checkpoints_data_and_smoke(tmp_path):
    project = tmp_path / "project"
    artifacts = project / "artifacts"
    run = artifacts / "runs" / "seed-42" / "coupled" / "fedex_lora"
    run.mkdir(parents=True)
    (run / "summary.json").write_text('{"ok": true}\n')
    (run / "rounds.json").write_text("[]\n")
    (run / "adapter_state.pt").write_bytes(b"large-checkpoint")
    (artifacts / "controlled_examples.jsonl").write_text("large generated data\n")
    smoke = artifacts / "smoke-runs"
    smoke.mkdir()
    (smoke / "summary.json").write_text('{"smoke": true}\n')
    (project / "configs").mkdir()
    (project / "configs" / "experiment.yaml").write_text("seed: 42\n")
    (project / "pyproject.toml").write_text("[project]\nname='test'\n")
    output = tmp_path / "results.zip"
    script = Path(__file__).parents[1] / "scripts" / "package_results.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project",
            str(project),
            "--artifacts",
            str(artifacts),
            "--output",
            str(output),
        ],
        check=True,
    )

    with ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "artifacts/runs/seed-42/coupled/fedex_lora/summary.json" in names
    assert "artifacts/runs/seed-42/coupled/fedex_lora/rounds.json" in names
    assert "provenance/configs/experiment.yaml" in names
    assert "provenance/git-commit.txt" in names
    assert not any(name.endswith(".pt") for name in names)
    assert not any("controlled_examples" in name for name in names)
    assert not any("smoke" in name for name in names)
