#!/bin/bash
# Incremental pipeline launcher. It records: unified input -> labels and
# historical-only prompts -> historical-only embeddings -> zero-copy merge ->
# all-classifier/seven-representation classification.
#SBATCH --job-name=cninfo-short-mask-pipeline
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/team/llm_return/logs/slurm_cninfo_short_mask_pipeline/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_cninfo_short_mask_pipeline/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/team/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?pipeline must use scripts/task_tracker.py}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
CURRENT_INPUT=${CURRENT_INPUT:-data/processed/cleaned/cninfo_announcements_prompt_inputs.jsonl}
CLEAN_INPUT=${CLEAN_INPUT:-data/processed/cleaned/cninfo_announcements_2010_2026_final.jsonl}
EXISTING_PANEL=${EXISTING_PANEL:-data/processed/cninfo_full_classification_panel.parquet}
EXISTING_ROWS=${EXISTING_ROWS:-350577}
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
UNIFIED_INPUT=${UNIFIED_INPUT:-data/processed/cleaned/cninfo_announcements_prompt_inputs_2010_2026.jsonl}
HISTORICAL_INPUT=${HISTORICAL_INPUT:-data/processed/cleaned/cninfo_announcements_prompt_inputs_2010_2017_increment.jsonl}
UNIFIED_SUMMARY=${UNIFIED_SUMMARY:-reports/cninfo_unified_classification_input_2010_2026.json}
BUNDLE_DIR=${BUNDLE_DIR:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017}
INPUT_ROOT=${INPUT_ROOT:-${BUNDLE_DIR}/prompt_v3_fixed_parts}
HISTORICAL_EMBEDDING_ROOT=${HISTORICAL_EMBEDDING_ROOT:-data/processed/pooled_embeddings_increment_2010_2017_v1}
EXISTING_EMBEDDING_ROOT=${EXISTING_EMBEDDING_ROOT:-data/processed/pooled_embeddings_v3}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-data/processed/pooled_embeddings_2010_2026_v1}
SUMMARY=${SUMMARY:-reports/cninfo_prompt_bundle_increment_2010_2017_audit.json}
CLASSIFICATION_CONCURRENCY=${CLASSIFICATION_CONCURRENCY:-16}
EMBEDDING_CONCURRENCY=${EMBEDDING_CONCURRENCY:-16}
SORT_CHUNK_ROWS=${SORT_CHUNK_ROWS:-2000}
SORT_TMP_ROOT=${SORT_TMP_ROOT:-}

tracked_sbatch() {
  "${PYTHON}" scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

mkdir -p logs/slurm_cninfo_short_mask_pipeline

unified_job=$(tracked_sbatch --parsable \
  --export="ALL,CURRENT=${CURRENT_INPUT},HISTORICAL=${CLEAN_INPUT},EXISTING_PANEL=${EXISTING_PANEL},STOCK_IDS_FROM=${EXISTING_PANEL},OUTPUT=${UNIFIED_INPUT},HISTORICAL_OUTPUT=${HISTORICAL_INPUT},SUMMARY=${UNIFIED_SUMMARY},SORT_CHUNK_ROWS=${SORT_CHUNK_ROWS},SORT_TMP_ROOT=${SORT_TMP_ROOT}" \
  scripts/slurm_build_cninfo_incremental_inputs.sbatch)

label_job=$(tracked_sbatch --parsable \
  --dependency="afterok:${unified_job}" \
  --export="ALL,INPUT=${UNIFIED_INPUT},PANEL=${PANEL}" \
  scripts/slurm_build_cninfo_all_labels_2010_2026.sbatch)

prompt_job=$(tracked_sbatch --parsable \
  --dependency="afterok:${unified_job}" \
  --export="ALL,INPUT=${HISTORICAL_INPUT},BUNDLE_DIR=${BUNDLE_DIR},SUMMARY=${SUMMARY},ROW_OFFSET=${EXISTING_ROWS:-350577}" \
  scripts/slurm_cninfo_prompt_bundle_incremental.sbatch)

embedding_job=$(tracked_sbatch --parsable \
  --dependency="afterok:${prompt_job}" \
  --array="0-63%${EMBEDDING_CONCURRENCY}" \
  --export="ALL,INPUT_ROOT=${INPUT_ROOT},OUTPUT_ROOT=${HISTORICAL_EMBEDDING_ROOT}" \
  scripts/slurm_short_pooled_embeddings_incremental.sbatch)

merge_job=$(tracked_sbatch --parsable \
  --dependency="afterok:${embedding_job}" \
  --export="ALL,EXISTING_ROOT=${EXISTING_EMBEDDING_ROOT},INCREMENTAL_ROOT=${HISTORICAL_EMBEDDING_ROOT},INCREMENTAL_INPUT=${HISTORICAL_INPUT},OUTPUT_ROOT=${EMBEDDING_ROOT}" \
  scripts/slurm_merge_cninfo_incremental_embeddings.sbatch)

classification_launcher_job=$(tracked_sbatch --parsable \
  --dependency="afterok:${label_job}:${merge_job}" \
  --export="ALL,PANEL=${PANEL},EMBEDDING_ROOT=${EMBEDDING_ROOT},CLASSIFICATION_CONCURRENCY=${CLASSIFICATION_CONCURRENCY}" \
  scripts/slurm_launch_cninfo_all_pooling_classification.sbatch)

echo "unified=${unified_job} label_panel=${label_job} historical_prompt=${prompt_job} historical_embeddings=${embedding_job} merged_embeddings=${merge_job} classification_launcher=${classification_launcher_job}"
