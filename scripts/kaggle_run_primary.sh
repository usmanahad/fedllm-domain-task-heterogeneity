#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
REGIME="${2:-coupled}"
AGGREGATION="${3:-fedex_lora}"
CONFIG="${CONFIG:-configs/controlled_qwen_p100.yaml}"
EXAMPLES="${EXAMPLES:-artifacts/controlled_examples.jsonl}"
CLIENTS="${CLIENTS:-16}"
PARTITION_SEED="${PARTITION_SEED:-42}"
DATA_TAG="${DATA_TAG:-legacy-v1}"
EXPECTED_EXAMPLES_SHA256="${EXPECTED_EXAMPLES_SHA256:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs}"
PARTITIONS_ROOT="${PARTITIONS_ROOT:-artifacts}"
PARTITIONS="${PARTITIONS_ROOT}/partitions-${DATA_TAG}-pseed-${PARTITION_SEED}-${REGIME}.json"
OUTPUT="${OUTPUT_ROOT}/${DATA_TAG}/partition-seed-${PARTITION_SEED}/seed-${SEED}/${REGIME}/${AGGREGATION}"

export HF_HOME="${HF_HOME:-/kaggle/working/hf-cache}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [[ ! -f "${EXAMPLES}" ]]; then
  fedllm-heterogeneity build-data \
    --config "${CONFIG}" \
    --output "${EXAMPLES}" \
    --cache-dir "${HF_HOME}/datasets"
fi

if [[ ! -f "${PARTITIONS}" ]]; then
  fedllm-heterogeneity build-partitions \
    --examples "${EXAMPLES}" \
    --regime "${REGIME}" \
    --clients "${CLIENTS}" \
    --seed "${PARTITION_SEED}" \
    --output "${PARTITIONS}"
fi

AUDIT_ARGS=(
  --examples "${EXAMPLES}"
  --partitions "${PARTITIONS}"
  --expected-regime "${REGIME}"
  --expected-partition-seed "${PARTITION_SEED}"
  --strict
)
if [[ -n "${EXPECTED_EXAMPLES_SHA256}" ]]; then
  AUDIT_ARGS+=(--expected-examples-sha256 "${EXPECTED_EXAMPLES_SHA256}")
fi
fedllm-heterogeneity audit "${AUDIT_ARGS[@]}"

fedllm-heterogeneity run-local \
  --config "${CONFIG}" \
  --examples "${EXAMPLES}" \
  --partitions "${PARTITIONS}" \
  --aggregation "${AGGREGATION}" \
  --seed "${SEED}" \
  --device cuda \
  --output "${OUTPUT}"

echo "Completed: ${OUTPUT}/summary.json"
