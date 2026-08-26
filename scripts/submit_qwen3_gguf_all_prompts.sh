#!/bin/bash
set -euo pipefail
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
ROOT=/home/gaozh/llm_return
SINA_INPUT=/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/inputs
CNINFO_INPUT=/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/cninfo_shards_1000
OUTPUT_ROOT=/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/qwen3_gguf_prompt_mean
GPU_NODELIST=${GPU_NODELIST:-c018-10gpu,c019-10gpu}
GPU_CANDIDATES=${GPU_CANDIDATES:-0,2,3}

submit_one() {
  local dataset=$1 variant=$2 prompt_id=$3 prompt_text=$4 shards=$5 input_root=$6
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "$TASK_RECORD_ID" -- sbatch \
    --nodelist="${GPU_NODELIST}" \
    --array="0-$((shards - 1))%3" \
    --export="ALL,DATASET=${dataset},VARIANT=${variant},PROMPT_ID=${prompt_id},PROMPT_TEXT=${prompt_text},SHARDS=${shards},INPUT_ROOT=${input_root},OUTPUT_ROOT=${OUTPUT_ROOT},GPU_CANDIDATES=${GPU_CANDIDATES}" \
    scripts/slurm_qwen3_gguf_prompt_mean.sbatch
}

prompts=(
  'future_return|分析股票未来收益。'
  'return|分析股票收益。'
  'loss|分析股票亏损。'
  'excess_return|分析股票超额收益。'
  'profit|分析股票盈利。'
)
for item in "${prompts[@]}"; do
  IFS='|' read -r id text <<< "$item"
  for variant in short masked_short; do
    submit_one sina "$variant" "$id" "$text" 64 "$SINA_INPUT"
    submit_one cninfo "$variant" "$id" "$text" 1000 "$CNINFO_INPUT"
  done
done
echo 'submitted all prompt/dataset/variant arrays'