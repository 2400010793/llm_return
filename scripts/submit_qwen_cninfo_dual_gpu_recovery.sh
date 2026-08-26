#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/gaozh/llm_return
cd "${REPO_ROOT}"

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name qwen-cninfo-dual-gpu-recovery \
    --purpose "Resume Qwen CNINFO prompt embeddings on c014/c018/c019 with fixed GPU sequences and disjoint shard arrays" \
    --snapshot-from-task 20260824T024110Z-qwen-cninfo-dual-gpu-recovery-57979 \
    --input /mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/cninfo_shards_1000 \
    --output /mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/qwen3_gguf_prompt_mean \
    --related-file scripts/slurm_qwen3_gguf_prompt_mean.sbatch \
    --related-file scripts/submit_qwen_cninfo_dual_gpu_recovery.sh \
    --tag qwen --tag cninfo --tag dual-gpu --tag memory-safe --seed 42 \
    -- bash "$0" "$@"
fi

PYTHON=/home/gaozh/llm_return/.venv/bin/python
TRACKER=("${PYTHON}" scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" --)
SCRIPT=scripts/slurm_qwen3_gguf_prompt_mean.sbatch
INPUT_ROOT=/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/cninfo_shards_1000
OUTPUT_ROOT=/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/qwen3_gguf_prompt_mean

missing_for_mod() {
  local prompt_id=$1 variant=$2 mod=$3
  local ids=() shard marker
  for ((shard=mod; shard<1000; shard+=3)); do
    marker="${OUTPUT_ROOT}/cninfo/${prompt_id}/${variant}/shard-${shard}/COMPLETED"
    [[ -f "${marker}" ]] || ids+=("${shard}")
  done
  local IFS=,
  printf '%s' "${ids[*]}"
}

submit_one() {
  local node=$1 ids=$2 candidates=$3 throttle=$4 variant=$5 prompt_id=$6 prompt_text=$7
  "${TRACKER[@]}" sbatch --parsable --nodelist="${node}" --array="${ids}%${throttle}" \
    --export="ALL,DATASET=cninfo,VARIANT=${variant},PROMPT_ID=${prompt_id},PROMPT_TEXT=${prompt_text},SHARDS=1000,INPUT_ROOT=${INPUT_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},GPU_CANDIDATES=${candidates},GPU_SLOTS_PER_DEVICE=2,GPU_BIND_MODE=sequence,TASK_RECORD_ID=${TASK_RECORD_ID}" \
    "${SCRIPT}"
}

for spec in \
  'future_return|分析股票未来收益。' \
  'return|分析股票收益。' \
  'profit|分析股票盈利。' \
  'excess_return|分析股票超额收益。' \
  'loss|分析股票亏损。'; do
  IFS='|' read -r prompt_id prompt_text <<< "${spec}"
  for variant in short masked_short; do
    ids=$(missing_for_mod "${prompt_id}" "${variant}" 0)
    [[ -n "${ids}" ]] && submit_one c014-epyc7763 "${ids}" 2,3,4,6,7,8 3 "${variant}" "${prompt_id}" "${prompt_text}"
    ids=$(missing_for_mod "${prompt_id}" "${variant}" 1)
    [[ -n "${ids}" ]] && submit_one c018-10gpu "${ids}" 0,1,2,3,4,8,9 2 "${variant}" "${prompt_id}" "${prompt_text}"
    ids=$(missing_for_mod "${prompt_id}" "${variant}" 2)
    [[ -n "${ids}" ]] && submit_one c019-10gpu "${ids}" 2,7,8,9 2 "${variant}" "${prompt_id}" "${prompt_text}"
  done
done

echo "submitted only missing CNINFO shards: modulo-3 split across c014/c018/c019 with fixed GPU sequences"
