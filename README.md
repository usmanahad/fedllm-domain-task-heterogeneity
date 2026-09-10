# Federated LLM Domain–Task Heterogeneity

This repository turns the exploratory FlowerTune notebooks into a reproducible
experiment package. It keeps the four FlowerTune corpora while separating two
questions that the original natural setup confounds:

1. What changes when clients have different **subject domains** but perform the
   same language-modeling task?
2. What changes when clients perform different **tasks**, independently of the
   domain of their text?

The original notebooks are retained as provenance:

- `fedavg-flowetune-phase1.ipynb`
- `fedavg-flowertune-kaggle-2.ipynb`

## What is implemented

- Adapters for the four official FlowerTune training datasets.
- Deterministic, source-level train/validation/test selection.
- Open-vocabulary continuation and span-reconstruction transformations.
- IID, domain-only, task-only, and coupled client partitions over the identical
  example pool.
- Factor-wise FedAvg, FFA-LoRA validation, exact FedEx-LoRA residuals, and
  rank-truncated effective-update aggregation.
- Geometry diagnostics, functional transfer matrices, leave-one-client-out
  harm analysis, normalized recovery, equivalence intervals, and attribution-
  guided client weights.
- A dependency-light CLI for building/auditing manifests and testing the LoRA
  algebra before expensive runs.

The package deliberately keeps data preparation and aggregation independent of
Flower and Transformers. Unit tests therefore run on CPU without downloading a
model. The optional integrations are installed only for real experiments.

## Installation

```bash
python -m pip install -e '.[dev]'
pytest
```

For a Kaggle Tesla P100, do not use the default Kaggle PyTorch build. Follow
the exact pinned setup and smoke test in [`KAGGLE_P100.md`](KAGGLE_P100.md).

Python 3.10–3.12 is recommended for the full Flower/Transformers stack.

## Clone on Kaggle

Once this repository is published on GitHub, a Kaggle notebook can clone it
directly into the writable workspace:

```bash
cd /kaggle/working
git clone https://github.com/usmanahad/fedllm-domain-task-heterogeneity.git Fred
cd Fred
```

Then follow the pinned P100 setup in [`KAGGLE_P100.md`](KAGGLE_P100.md), starting
at the PyTorch installation step. The copy-from-`/kaggle/input` step is not
needed when the project is cloned.

## Prepare the controlled dataset

```bash
fedllm-heterogeneity build-data \
  --config configs/controlled_qwen.yaml \
  --output artifacts/controlled_examples.jsonl

fedllm-heterogeneity build-partitions \
  --examples artifacts/controlled_examples.jsonl \
  --regime coupled \
  --clients 16 \
  --seed 42 \
  --output artifacts/partitions-coupled.json

fedllm-heterogeneity audit \
  --examples artifacts/controlled_examples.jsonl \
  --partitions artifacts/partitions-coupled.json
```

`build-data` downloads the configured Hugging Face datasets. It selects source
rows before creating task views, so a source can never cross data splits. The
manifest records hashes, dataset revisions when available, and every transform
seed.

## Validate aggregation before training

```bash
fedllm-heterogeneity lora-demo --seed 42
```

The command constructs toy LoRA clients and shows:

- the error introduced by independently averaging `A` and `B`;
- the zero effective-model error obtained with a FedEx residual;
- the reconstruction error caused by a requested SVD rank.

## Experiment tracks

`configs/controlled_qwen.yaml` defines the primary 4-domain × 2-task experiment.
`configs/natural_qwen.yaml` defines the original instruction-response tasks with
four clients per domain. The natural track is intentionally treated as a
domain/task-confounded external-validity experiment.

Heavy training is exposed through small reusable components rather than hidden
inside a notebook. `fedllm_heterogeneity.training` contains the PEFT model and
client-training helpers, while `fedllm_heterogeneity.flower_app` provides the
Flower-facing client-data contract and aggregation strategy adapter.

Generate the complete three-seed command matrix without starting any jobs:

```bash
PYTHONPATH=src python scripts/print_run_matrix.py \
  --config configs/controlled_qwen.yaml \
  --seeds 42 43 44
```

For one local sequential run:

```bash
fedllm-heterogeneity run-local \
  --config configs/controlled_qwen.yaml \
  --examples artifacts/controlled_examples.jsonl \
  --partitions artifacts/partitions-coupled.json \
  --aggregation fedex_lora \
  --seed 42 \
  --output artifacts/runs/seed-42/coupled/fedex_lora
```

The equivalent Flower simulation is configured in `pyproject.toml`:

```bash
flwr run --run-config \
  "partitions='artifacts/partitions-coupled.json' aggregation='fedex_lora' seed=42"
```

Base, centralized, and local-only references use `run-reference`. Once all
seeds are present, produce normalized recovery, factorial contrasts, and the
equivalence decision with:

```bash
fedllm-heterogeneity analyze \
  --runs artifacts/runs \
  --output artifacts/analysis.json
```

For native FlowerTune evaluation, first export the exact adapter plus FedEx
base residual as a normal Hugging Face checkpoint, then invoke the evaluation
suite. Code evaluation is blocked unless `--allow-code-execution` is explicitly
provided and should only run in an isolated Kaggle runtime.

## Interpretation guardrails

- Raw response NLL is reported per cell; it is not averaged across unrelated
  native tasks as if the scales were directly comparable.
- Gradient conflict is measured at a common checkpoint. Multi-step local model
  deltas are labelled as updates, not gradients.
- LoRA geometry is computed from the induced `BA` matrices, not only from the
  factor coordinates.
- “Equivalent to IID” requires the configured confidence interval to clear the
  non-inferiority margin; a non-significant difference is not enough.
- ProToken-style attribution is not automatically interpreted as harm. Client
  weighting is enabled only after attribution is validated against measured
  leave-one-client-out effects.

Selected diagnostic rounds save the individual client adapter states needed by
the official [ProToken artifact](https://github.com/ahmayun/protoken). ProToken
itself is not vendored: its reference reproduction targets an A100-class Linux
machine with substantially more RAM/storage than a normal Kaggle session. This
package supplies the client checkpoints and implements the causal validation
and guarded reweighting stage that ProToken does not provide.
