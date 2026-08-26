#!/bin/bash
# Attach triple-concat Ridge regressions to existing embedding parent arrays.
set -euo pipefail
cd /home/gaozh/llm_return
if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name pooled-embedding-regression \
    --purpose "Attach aligned pooled-embedding Ridge regression jobs" \
    --input data/processed/cninfo_full_classification_panel.parquet \
    --output reports/regression/pooled_embeddings \
    --related-file scripts/slurm_launch_completed_embedding_regression.sbatch \
    --related-file scripts/slurm_pooled_embedding_regression.sbatch \
    -- bash "$0" "$@"
fi
tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}
SHORT_EMBED_JOB=${SHORT_EMBED_JOB:-3847458}
LONG_EMBED_JOB=${LONG_EMBED_JOB:-3847506}
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel.parquet}
[[ -s "${PANEL}" ]] || { echo "missing panel: ${PANEL}" >&2; exit 2; }
mkdir -p logs/slurm_pooled_regression logs/slurm_pooled_regression_launcher configs/generated

/home/gaozh/llm_return/.venv/bin/python -m py_compile \
  scripts/run_pooled_embedding_regression.py scripts/build_pooled_regression_manifest.py
bash -n scripts/slurm_pooled_embedding_regression.sbatch scripts/slurm_launch_completed_embedding_regression.sbatch

submit_group() {
  local embed_job=$1 model=$2 variant_group=$3
  local job
  job=$(tracked_sbatch --parsable \
    --dependency="afterok:${embed_job}" \
    --export="ALL,MODEL=${model},VARIANT_GROUP=${variant_group},PANEL=${PANEL}" \
    scripts/slurm_launch_completed_embedding_regression.sbatch)
  echo "attached regression launcher=${job} embedding=${embed_job} model=${model} group=${variant_group}"
}

submit_group "${SHORT_EMBED_JOB}" roberta short
submit_group "${SHORT_EMBED_JOB}" bge_m3 short
submit_group "${LONG_EMBED_JOB}" roberta long
submit_group "${LONG_EMBED_JOB}" bge_m3 long