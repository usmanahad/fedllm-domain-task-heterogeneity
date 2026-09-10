#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
REGIME="${2:-coupled}"
AGGREGATION="${3:-fedex_lora}"
CONFIG="${CONFIG:-configs/controlled_qwen_p100.yaml}"
EXAMPLES="${EXAMPLES:-artifacts/controlled_examples.jsonl}"
CLIENTS="${CLIENTS:-16}"
PARTITION_SEED="${PARTITION_SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs}"
PARTITIONS_ROOT="${PARTITIONS_ROOT:-artifacts}"
PARTITIONS="${PARTITIONS_ROOT}/partitions-${REGIME}.json"
OUTPUT="${OUTPUT_ROOT}/seed-${SEED}/${REGIME}/${AGGREGATION}"

export HF_HOME="${HF_HOME:-/kaggle/working/hf-cache}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

fedllm-heterogeneity audit \
  --examples "${EXAMPLES}" \
  --partitions "${PARTITIONS}"

fedllm-heterogeneity run-local \
  --config "${CONFIG}" \
  --examples "${EXAMPLES}" \
  --partitions "${PARTITIONS}" \
  --aggregation "${AGGREGATION}" \
  --seed "${SEED}" \
  --device cuda \
  --output "${OUTPUT}"

echo "Completed: ${OUTPUT}/summary.json"
