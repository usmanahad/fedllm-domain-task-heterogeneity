# Locked Kaggle P100 pilot

This is the next experiment after the CPU-only audit. It runs one optimization
seed (`42`), two partition regimes (`iid`, `domain_only`), and two LoRA
aggregation methods (`factor_fedavg`, `fedex_lora`): four federated runs total.
It also evaluates the pretrained base once and trains one centralized reference.

The public repository contains the code and frozen data-selection anchor. The
upload-only `kaggle-p100-runner.ipynb` stays local and is intentionally excluded
from Git.

## What each arm tests

- **IID:** every one of the 16 clients receives the same balanced mixture of all
  four domains and both open-vocabulary tasks.
- **Domain-only:** four clients receive only general-language sources, four only
  finance, four only medicine, and four only code. Every client still mixes both
  tasks. The data pool and number of examples are identical to IID.
- **Factor FedAvg:** independently averages the trainable LoRA `A` and `B`
  factors. This is common practice, but is not the exact average of the induced
  weight updates `BA`.
- **FedEx-LoRA:** preserves the exact weighted average of the induced LoRA
  updates through a residual. It is the aggregation-correctness control, not an
  assumption that it must perform better.

This pilot is a performance screen. Expensive gradient, transfer-matrix, and
leave-one-client-out diagnostics are disabled. They will be timed on a selected
follow-up run only if the pilot justifies them.

## P100-safe setup

1. Upload `kaggle-p100-runner.ipynb` as a Kaggle Notebook.
2. Select **GPU P100** and turn **Internet on**.
3. Start a fresh session and choose **Run all**.

The notebook performs the following operations in order:

1. Clones or fast-forwards
   `https://github.com/usmanahad/fedllm-domain-task-heterogeneity.git`.
2. Replaces Kaggle's incompatible CUDA 12.8 PyTorch packages with PyTorch
   2.5.1 CUDA 12.1, whose wheel supports the P100's Pascal architecture.
3. Installs `requirements-kaggle-p100.txt` and the project without resolving
   additional dependencies.
4. Runs the CUDA fp16 preflight, the CPU test suite, the LoRA algebra test, and
   an optional tiny end-to-end smoke run.
5. Builds and strictly audits the clean-v2 controlled pool and both partitions.
6. Runs the two references and four federated arms sequentially on one GPU.
7. Creates one compact ZIP per federated arm in
   `/kaggle/working/fedllm-result-zips`.

Do not use bf16 or FlashAttention 2 on a P100. The pilot uses
Qwen2.5-0.5B-Instruct, fp16, batch size 2, sequence length 512, and sequential
clients so only one model is resident on the 16 GiB GPU. If the smoke test runs
out of memory, set both training and evaluation batch sizes to 1 in a copy of
the config; apply the same change to all four arms.

## Exact manual commands

The notebook is the recommended path. These commands reproduce its scientific
steps after the pinned environment has been installed:

```bash
cd /kaggle/working/Fred
export HF_HOME=/kaggle/working/hf-cache
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

fedllm-heterogeneity build-data \
  --config configs/controlled_qwen_p100_clean_v2.yaml \
  --output artifacts/controlled_examples_clean_v2.jsonl \
  --cache-dir /kaggle/working/hf-cache/datasets

for regime in iid domain_only; do
  fedllm-heterogeneity build-partitions \
    --examples artifacts/controlled_examples_clean_v2.jsonl \
    --regime "$regime" \
    --clients 16 \
    --seed 42 \
    --output "artifacts/partitions-clean-v2-pseed-42-${regime}.json"

  fedllm-heterogeneity audit \
    --examples artifacts/controlled_examples_clean_v2.jsonl \
    --partitions "artifacts/partitions-clean-v2-pseed-42-${regime}.json" \
    --expected-regime "$regime" \
    --expected-partition-seed 42 \
    --expected-examples-sha256 \
      ab07a92891d0ccaeaa0a77ed570f5fbcf6c31395e3c1c5b79185e102b30d6418 \
    --strict
done
```

Run the references once:

```bash
COMMON_ROOT=artifacts/runs/clean-v2/partition-seed-42/seed-42

fedllm-heterogeneity run-reference \
  --config configs/controlled_qwen_p100_clean_v2.yaml \
  --examples artifacts/controlled_examples_clean_v2.jsonl \
  --arm base --seed 42 --device cuda \
  --output "${COMMON_ROOT}/base"

fedllm-heterogeneity run-reference \
  --config configs/controlled_qwen_p100_clean_v2.yaml \
  --examples artifacts/controlled_examples_clean_v2.jsonl \
  --arm centralized --seed 42 --device cuda \
  --output "${COMMON_ROOT}/centralized"
```

Run the four federated arms, one after another:

```bash
for regime in iid domain_only; do
  for aggregation in factor_fedavg fedex_lora; do
    CONFIG=configs/controlled_qwen_p100_clean_v2.yaml \
    EXAMPLES=artifacts/controlled_examples_clean_v2.jsonl \
    DATA_TAG=clean-v2 \
    PARTITION_SEED=42 \
    EXPECTED_EXAMPLES_SHA256=ab07a92891d0ccaeaa0a77ed570f5fbcf6c31395e3c1c5b79185e102b30d6418 \
    bash scripts/kaggle_run_primary.sh 42 "$regime" "$aggregation"
  done
done
```

Expected output directories are:

```text
artifacts/runs/clean-v2/partition-seed-42/seed-42/iid/factor_fedavg/
artifacts/runs/clean-v2/partition-seed-42/seed-42/iid/fedex_lora/
artifacts/runs/clean-v2/partition-seed-42/seed-42/domain_only/factor_fedavg/
artifacts/runs/clean-v2/partition-seed-42/seed-42/domain_only/fedex_lora/
```

## Data-integrity gate

The clean-v2 config pins all four Hugging Face dataset revisions and uses
`configs/controlled_source_splits_v1.json` as the source-row anchor. The locally
verified pool has:

- 7,168 task examples derived from 3,584 unique sources;
- exactly 512 train, 128 validation, and 256 test examples per domain-task cell;
- no source overlap across splits and no duplicate example IDs;
- no known FlowerTune finance or medical instruction boilerplate in prompts or
  targets;
- SHA-256 `ab07a92891d0ccaeaa0a77ed570f5fbcf6c31395e3c1c5b79185e102b30d6418`.

General and code preserve 100% of anchored sources. Medical preserves about
94%, while finance preserves about 40% because removing its long fixed
sentiment instruction makes many statements shorter than the locked 32-token
minimum. Replacements are deterministic and source-local; the same clean pool
is used for IID and domain-only.

The Kaggle manifest hash must match the value above. Stop rather than train if
it does not.

## Downloads and recovery

Each ZIP contains JSON measurements, tables, the selected data manifest, pinned
configs, split anchor, dependency pins, reference summaries, and Git revision.
Large adapters, FedEx residual tensors, generated datasets, and partitions are
excluded from the ZIP. The run directories retain final checkpoints because
they may be needed for native evaluation; save `/kaggle/working/Fred/artifacts`
as a private Kaggle Dataset before the session expires if those checkpoints
matter.

If a session stops, rerun the notebook after attaching or restoring the
artifacts. It skips an arm only when `summary.json` exists and `rounds.json`
contains all eight rounds. Set `FORCE_RERUN = True` only when an intentional
full rerun is required.
