#!/bin/bash
# Build the unified panel input and the historical-only prompt input.
# Existing 2018--2026 row order is validated and never sent to the embedding
# generator in the downstream incremental stages.
#SBATCH --job-name=cninfo-increment-inputs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_cninfo_increment_inputs/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_cninfo_increment_inputs/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?incremental input construction must use scripts/task_tracker.py}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
CURRENT=${CURRENT:-data/processed/cleaned/cninfo_announcements_prompt_inputs.jsonl}
HISTORICAL=${HISTORICAL:-data/processed/cleaned/cninfo_announcements_2010_2026_final.jsonl}
EXISTING_PANEL=${EXISTING_PANEL:-data/processed/cninfo_full_classification_panel.parquet}
STOCK_IDS_FROM=${STOCK_IDS_FROM:-${EXISTING_PANEL}}
OUTPUT=${OUTPUT:-data/processed/cleaned/cninfo_announcements_prompt_inputs_2010_2026.jsonl}
HISTORICAL_OUTPUT=${HISTORICAL_OUTPUT:-data/processed/cleaned/cninfo_announcements_prompt_inputs_2010_2017_increment.jsonl}
SUMMARY=${SUMMARY:-reports/cninfo_unified_classification_input_2010_2026.json}
SORT_CHUNK_ROWS=${SORT_CHUNK_ROWS:-2000}
SORT_TMP_ROOT=${SORT_TMP_ROOT:-${SLURM_TMPDIR:-/tmp}}

mkdir -p logs/slurm_cninfo_increment_inputs "$(dirname "${OUTPUT}")" "$(dirname "${SUMMARY}")"
mkdir -p "${SORT_TMP_ROOT}"
SORT_TMP_DIR=$(mktemp -d "${SORT_TMP_ROOT%/}/cninfo-unified-${SLURM_JOB_ID:-local}.XXXXXX")
trap 'rm -rf -- "${SORT_TMP_DIR}"' EXIT
export TMPDIR="${SORT_TMP_DIR}"
"${PYTHON}" -m py_compile scripts/build_cninfo_unified_classification_input.py
"${PYTHON}" scripts/build_cninfo_unified_classification_input.py \
  --current "${CURRENT}" \
  --historical "${HISTORICAL}" \
  --existing-panel "${EXISTING_PANEL}" \
  --stock-ids-from "${STOCK_IDS_FROM}" \
  --output "${OUTPUT}" \
  --historical-output "${HISTORICAL_OUTPUT}" \
  --summary "${SUMMARY}" \
  --sort-chunk-rows "${SORT_CHUNK_ROWS}" \
  --sort-temp-dir "${SORT_TMP_DIR}"

[[ -s "${OUTPUT}" && -s "${HISTORICAL_OUTPUT}" ]] || {
  echo "unified or historical incremental input is empty" >&2
  exit 2
}
