#!/bin/bash
set -euo pipefail

: "${TASK_RECORD_ID:?run through scripts/task_tracker.py}"
: "${START_DEPENDENCY:?set to the current Sina Qwen array dependency}"

REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
INPUT_ROOT=${INPUT_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/cninfo_shards_1000}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/qwen3_gguf_prompt_mean}
GPU_NODES=${GPU_NODES:-c014-epyc7763:c018-10gpu}
GPU_CANDIDATES=${GPU_CANDIDATES:-0:1:2:3:4:5:6:7:8:9}
ARRAY_THROTTLE=${ARRAY_THROTTLE:-2}
SHARDS=${SHARDS:-1000}

tracked() {
  "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/task_tracker.py" \
    child-submit --task-id "${TASK_RECORD_ID}" -- "$@"
}

prompts=(
  'future_return|分析股票未来收益。'
  'return|分析股票收益。'
  'profit|分析股票盈利。'
  'loss|分析股票亏损。'
  'excess_return|分析股票超额收益。'
)

job_ids=()
IFS=':' read -r -a gpu_nodes <<< "${GPU_NODES}"
(( ${#gpu_nodes[@]} > 0 )) || { echo 'GPU_NODES is empty' >&2; exit 2; }
combination_index=0
for item in "${prompts[@]}"; do
  IFS='|' read -r prompt_id prompt_text <<< "${item}"
  for variant in short masked_short; do
    gpu_node=${gpu_nodes[$((combination_index % ${#gpu_nodes[@]}))]}
    job_id=$(tracked sbatch --parsable \
      --dependency="afterany:${START_DEPENDENCY}" \
      --nodes=1 --nodelist="${gpu_node}" \
      --array="0-$((SHARDS - 1))%${ARRAY_THROTTLE}" \
      --export="ALL,DATASET=cninfo,VARIANT=${variant},PROMPT_ID=${prompt_id},PROMPT_TEXT=${prompt_text},SHARDS=${SHARDS},INPUT_ROOT=${INPUT_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},GPU_SLOTS_PER_DEVICE=1,GPU_CANDIDATES=${GPU_CANDIDATES}" \
      "${REPO_ROOT}/scripts/slurm_qwen3_gguf_prompt_mean.sbatch")
    job_ids+=("${job_id}")
    echo "submitted prompt=${prompt_id} variant=${variant} node=${gpu_node} job=${job_id}"
    combination_index=$((combination_index + 1))
  done
done

printf 'cninfo_recovery_jobs=%s\n' "$(IFS=:; echo "${job_ids[*]}")"
