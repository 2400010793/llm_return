#!/bin/bash
# Attach follow-up classification jobs to the existing embedding arrays without
# changing, cancelling, or resubmitting those arrays.
set -euo pipefail
cd /home/gaozh/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name pooled-embedding-classification \
    --purpose "Attach aligned pooled-embedding classification jobs" \
    --input data/processed/cninfo_full_classification_panel.parquet \
    --output reports/classification/pooled_embeddings \
    --related-file scripts/slurm_launch_completed_embedding_classification.sbatch \
    --related-file scripts/slurm_pooled_embedding_classification.sbatch \
    -- bash "$0" "$@"
fi
tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

SHORT_EMBED_JOB=${SHORT_EMBED_JOB:-3847458}
LONG_EMBED_JOB=${LONG_EMBED_JOB:-3847506}
PHASES=${PHASES:-screening+aggregation}
INCLUDE_PROMPT_TOKENS=${INCLUDE_PROMPT_TOKENS:-0}
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel.parquet}
PANEL_JOB=${PANEL_JOB:-}
mkdir -p logs/slurm_full_panel logs/slurm_pooled_cls_launcher logs/slurm_pooled_cls configs/generated

# Cheap checks before any submission.
/home/gaozh/llm_return/.venv/bin/python -m py_compile \
  scripts/build_cninfo_full_classification_panel.py \
  scripts/build_pooled_classification_manifest.py \
  scripts/run_pooled_embedding_classification.py
bash -n \
  scripts/slurm_build_cninfo_full_classification_panel.sbatch \
  scripts/slurm_launch_completed_embedding_classification.sbatch \
  scripts/slurm_pooled_embedding_classification.sbatch

panel_dependency=""
if [[ ! -s "${PANEL}" ]]; then
  if [[ -n "${PANEL_JOB}" ]]; then
    panel_dependency=":${PANEL_JOB}"
    echo "reusing pending classification panel job=${PANEL_JOB}"
  else
    panel_job=$(tracked_sbatch --parsable scripts/slurm_build_cninfo_full_classification_panel.sbatch)
    panel_dependency=":${panel_job}"
    echo "submitted full classification panel job=${panel_job}"
  fi
else
  echo "reusing classification panel ${PANEL}"
fi

submit_group() {
  local embed_job=$1
  local model=$2
  local variant_group=$3
  # This cluster rejects dependencies listing future individual array
  # elements. Depend on the parent array instead; both model launchers start
  # immediately when the corresponding short/long array has fully succeeded.
  local dependency="afterok:${embed_job}${panel_dependency}"
  local launcher
  launcher=$(tracked_sbatch --parsable \
    --dependency="${dependency}" \
    --export="ALL,MODEL=${model},VARIANT_GROUP=${variant_group},PHASES=${PHASES},INCLUDE_PROMPT_TOKENS=${INCLUDE_PROMPT_TOKENS},PANEL=${PANEL}" \
    scripts/slurm_launch_completed_embedding_classification.sbatch)
  echo "attached launcher=${launcher} to embedding=${embed_job} model=${model} variant_group=${variant_group}"
}

# Even array tasks are RoBERTa; odd array tasks are BGE-M3. Each task writes
# both raw and masked variants for its shard.
submit_group "${SHORT_EMBED_JOB}" roberta short
submit_group "${SHORT_EMBED_JOB}" bge_m3 short
submit_group "${LONG_EMBED_JOB}" roberta long
submit_group "${LONG_EMBED_JOB}" bge_m3 long