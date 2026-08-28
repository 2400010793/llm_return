#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-20260812T080513Z-sina-prompt-regression-nonqwen-968042}
  exec .venv/bin/python scripts/task_tracker.py run \
    --name neutral-masked-short-roberta-bge-full-2010-2026 \
    --purpose "Full Sina and CNINFO neutral masked-short embeddings with RoBERTa and BGE-M3" \
    --input /data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/cleaned_full_2010_2026/sina_all_news_clean_expanded.parquet \
    --input data/processed/cleaned/cninfo_announcements_2010_2026_final.jsonl \
    --input data/processed/cninfo_full_classification_panel_2010_2026.parquet \
    --output /data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/neutral_masked_short_v1 \
    --output reports/neutral_masked_short_v1 \
    --related-file scripts/build_neutral_masked_inputs.py \
    --related-file scripts/run_neutral_masked_embeddings.py \
    --related-file scripts/audit_neutral_prompt_tokens.py \
    --related-file scripts/slurm_neutral_masked_embeddings.sbatch \
    --related-file configs/prompts/neutral_masked_short_v1.json \
    --snapshot-from-task "${SNAPSHOT_FROM_TASK}" \
    --tag neutral --tag masked-short --tag roberta --tag bge-m3 --tag full-history --seed 42 \
    -- bash "$0" "$@"
fi

ROOT=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/neutral_masked_short_v1
INPUT_ROOT=${INPUT_ROOT:-${ROOT}/inputs}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/embeddings}
CONFIG=${CONFIG:-/data/alpha_team2/shares/llm_return/configs/prompts/neutral_masked_short_v1.json}
# Exclude nodes with known DOWN/DRAIN reasons (reboot, NIC, slurmd spool,
# epilog failures). Healthy nodes remain scheduler-selected for each array.
EXCLUDE_NODES=${EXCLUDE_NODES:-c003-epyc9755,c005-epyc9755,c006-epyc9755,c010-epyc9755,c011-epyc9755,c118-epyc9575f,v126-epyc9575f,v127-epyc9575f,v128-epyc9575f,v129-epyc9575f,v134-epyc9575f}
SINA_CLEAN=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/cleaned_full_2010_2026/sina_all_news_clean_expanded.parquet
SINA_PANEL=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/classification/sina_full_classification_panel.parquet
CNINFO_CLEAN=/data/alpha_team2/shares/llm_return/data/processed/cleaned/cninfo_announcements_2010_2026_final.jsonl
CNINFO_PANEL=/data/alpha_team2/shares/llm_return/data/processed/cninfo_full_classification_panel_2010_2026.parquet

mkdir -p "${INPUT_ROOT}" reports/neutral_masked_short_v1
if [[ ! -s "${INPUT_ROOT}/sina/manifest.json" ]]; then
  .venv/bin/python scripts/build_neutral_masked_inputs.py --dataset sina \
    --clean "${SINA_CLEAN}" --panel "${SINA_PANEL}" \
    --output "${INPUT_ROOT}/sina" --shards 256
fi
if [[ ! -s "${INPUT_ROOT}/cninfo/manifest.json" ]]; then
  bash scripts/build_neutral_cninfo_bundle_inputs.sh "${INPUT_ROOT}/cninfo"
fi

.venv/bin/python scripts/audit_neutral_prompt_tokens.py --config "${CONFIG}" --model roberta \
  --output reports/neutral_masked_short_v1/roberta_preflight.json
.venv/bin/python scripts/audit_neutral_prompt_tokens.py --config "${CONFIG}" --model bge_m3 \
  --output reports/neutral_masked_short_v1/bge_m3_preflight.json

tracked() {
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- "$@"
}

jobs=()
for dataset in sina cninfo; do
  for model in roberta bge_m3; do
    job=$(tracked sbatch --parsable \
      --array=0-255%256 \
      --exclude="${EXCLUDE_NODES}" \
      --export=ALL,DATASET=${dataset},MODEL=${model},REPO_ROOT=/data/alpha_team2/shares/llm_return,INPUT_ROOT=${INPUT_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},CONFIG=${CONFIG} \
      scripts/slurm_neutral_masked_embeddings.sbatch)
    jobs+=("${dataset}/${model}:${job}")
  done
done
printf '%s\n' "${jobs[@]}"
