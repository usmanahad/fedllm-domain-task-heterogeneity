# Run this project on a Kaggle Tesla P100

The commands below target one 16 GiB Tesla P100. The recommended execution
path is `run-local`: it simulates the 16 clients sequentially on one GPU and
therefore does not keep multiple copies of the language model in VRAM.

## 0. Create the notebook

1. Upload the local `kaggle-p100-runner.ipynb` file as a Kaggle Notebook, or
   create a blank notebook and use the cells below.
2. In **Notebook options**, select **Accelerator: GPU P100**.
3. Turn **Internet on**. It is required for GitHub, pip, and Hugging Face.
4. Start with a fresh session and run the following cells in order.

## 1. Clone the public repository into the writable directory

```python
%cd /kaggle/working
!git clone https://github.com/usmanahad/fedllm-domain-task-heterogeneity.git Fred
%cd /kaggle/working/Fred
```

## 2. Replace Kaggle's incompatible CUDA 12.8 PyTorch wheel

Run this cell before importing PyTorch. It deliberately restarts the Python
kernel when installation finishes; wait for Kaggle to reconnect before running
the next cell.

```python
%cd /kaggle/working/Fred
!python -m pip uninstall -y torch torchvision torchaudio torchtext
!python -m pip install --no-cache-dir --force-reinstall \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

import os
os._exit(0)
```

The CUDA 12.1 wheel is intentional. It includes `sm_60` kernels for the Pascal
P100, whereas the current Kaggle CUDA 12.8 PyTorch image does not.

## 3. Install the pinned research environment

```python
%cd /kaggle/working/Fred
!python -m pip install --no-cache-dir -r requirements-kaggle-p100.txt
!python -m pip install --no-deps -e .
```

Do not replace these commands with `pip install -e '.[train,flower,eval]'` on a
P100: the loose extras can upgrade the deliberately pinned PyTorch and
Transformers versions.

## 4. Verify that a real P100 kernel runs

```python
%cd /kaggle/working/Fred
!nvidia-smi
!python scripts/kaggle_p100_preflight.py
!pytest
!fedllm-heterogeneity lora-demo --seed 42
```

The preflight must end in both `CUDA fp16 forward/backward: PASS` and `P100
preflight: PASS`. Merely seeing `torch.cuda.is_available() == True` is not
sufficient.

## 5. Run the small end-to-end smoke test

```python
%cd /kaggle/working/Fred
%env HF_HOME=/kaggle/working/hf-cache
%env TOKENIZERS_PARALLELISM=false
%env WANDB_DISABLED=true
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

!CONFIG=configs/smoke_p100.yaml \
  EXAMPLES=artifacts/smoke_examples.jsonl \
  CLIENTS=8 \
  PARTITIONS_ROOT=artifacts/smoke-partitions \
  OUTPUT_ROOT=artifacts/smoke-runs \
  bash scripts/kaggle_run_primary.sh 42 coupled fedex_lora
```

This downloads the four FlowerTune datasets and Qwen model, creates the eight
domain-task cells, audits the partition, trains one step for each of eight
clients, evaluates the cells, and writes:

```text
/kaggle/working/Fred/artifacts/smoke-runs/seed-42/coupled/fedex_lora/summary.json
```

Do not start the full experiment until this cell succeeds.

## 6. Build the full fixed example pool and all four partitions

Run once. All experimental runs reuse these exact files.

```python
%cd /kaggle/working/Fred
!fedllm-heterogeneity build-data \
  --config configs/controlled_qwen_p100.yaml \
  --output artifacts/controlled_examples.jsonl \
  --cache-dir /kaggle/working/hf-cache/datasets

!fedllm-heterogeneity build-partitions --examples artifacts/controlled_examples.jsonl --regime iid         --clients 16 --seed 42 --output artifacts/partitions-iid.json
!fedllm-heterogeneity build-partitions --examples artifacts/controlled_examples.jsonl --regime domain_only --clients 16 --seed 42 --output artifacts/partitions-domain_only.json
!fedllm-heterogeneity build-partitions --examples artifacts/controlled_examples.jsonl --regime task_only   --clients 16 --seed 42 --output artifacts/partitions-task_only.json
!fedllm-heterogeneity build-partitions --examples artifacts/controlled_examples.jsonl --regime coupled     --clients 16 --seed 42 --output artifacts/partitions-coupled.json

!fedllm-heterogeneity audit \
  --examples artifacts/controlled_examples.jsonl \
  --partitions artifacts/partitions-coupled.json
```

## 7. Run one primary experiment

Use one sequential job at a time. The three arguments are training seed,
partition regime, and aggregation method.

```python
%cd /kaggle/working/Fred
!bash scripts/kaggle_run_primary.sh 42 coupled fedex_lora
```

Other registered arms use the same command form:

```python
!bash scripts/kaggle_run_primary.sh 42 iid fedex_lora
!bash scripts/kaggle_run_primary.sh 42 domain_only fedex_lora
!bash scripts/kaggle_run_primary.sh 42 task_only fedex_lora
!bash scripts/kaggle_run_primary.sh 42 coupled factor_fedavg
!bash scripts/kaggle_run_primary.sh 42 coupled ffa_lora
!bash scripts/kaggle_run_primary.sh 42 coupled svd_effective
```

Repeat with seeds `43` and `44`. Keep the data and partition files fixed; only
the training seed changes. Generate the full command list with:

```python
!python scripts/print_run_matrix.py \
  --config configs/controlled_qwen_p100.yaml \
  --seeds 42 43 44 \
  --output-root artifacts/runs
```

The printed commands are a preregistered matrix, not a recommendation to run
all jobs in a single Kaggle session. Save `/kaggle/working/Fred/artifacts` as a
Kaggle output Dataset between sessions.

## 8. Reference arms and final analysis

```python
%cd /kaggle/working/Fred
!fedllm-heterogeneity run-reference \
  --config configs/controlled_qwen_p100.yaml \
  --examples artifacts/controlled_examples.jsonl \
  --arm base --seed 42 --device cuda \
  --output artifacts/runs/seed-42/base

!fedllm-heterogeneity run-reference \
  --config configs/controlled_qwen_p100.yaml \
  --examples artifacts/controlled_examples.jsonl \
  --arm centralized --seed 42 --device cuda \
  --output artifacts/runs/seed-42/centralized

!fedllm-heterogeneity run-reference \
  --config configs/controlled_qwen_p100.yaml \
  --examples artifacts/controlled_examples.jsonl \
  --partitions artifacts/partitions-coupled.json \
  --arm local_only --seed 42 --device cuda \
  --output artifacts/runs/seed-42/local-only

!fedllm-heterogeneity analyze \
  --runs artifacts/runs \
  --output artifacts/analysis.json
```

Repeat the three reference commands for seeds 43 and 44 before treating the
confidence intervals as final.

## Optional: Flower/Ray execution

The sequential runner above is scientifically equivalent for full client
participation and is safer on one P100. To exercise the Flower application
itself, install its separate pins and allow only one GPU client at a time:

```python
%cd /kaggle/working/Fred
!python -m pip install --no-cache-dir -r requirements-kaggle-p100-flower.txt
!flwr run --run-config \
  "experiment-config='configs/controlled_qwen_p100.yaml' examples='artifacts/controlled_examples.jsonl' partitions='artifacts/partitions-coupled.json' aggregation='fedex_lora' output='artifacts/flower-run' seed=42"
```

Do not set a fractional `num-gpus` value for this project on the P100: each
client constructs a model, and concurrent clients can exhaust 16 GiB VRAM.

## P100 operating constraints

- Use fp16, not bf16. The supplied configs and 4-bit compute path use fp16.
- Do not enable FlashAttention 2; the P100 is Pascal, not Ampere.
- Start with Qwen2.5-0.5B and `batch_size: 2`, `eval_batch_size: 2`, sequence
  length 512. The 1.5B QLoRA file is a later scale check, not the first run.
- If memory is still exhausted, set both batch sizes to 1. Do not change local
  steps or sequence length in only one experimental regime.
- `/kaggle/input` is read-only. All artifacts and Hugging Face caches must stay
  below `/kaggle/working`.
