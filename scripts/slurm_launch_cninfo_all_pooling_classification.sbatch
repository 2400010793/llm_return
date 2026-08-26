#!/bin/bash
# Build and submit the all-pooling one-day classification array after the
# all-article label panel and both short embedding variants are complete.
#SBATCH --job-name=launch-cninfo-all-pool
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_cninfo_all_pool_cls_launcher/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_cninfo_all_pool_cls_launcher/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?classification launcher must use scripts/task_tracker.py}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-data/processed/pooled_embeddings_2010_2026_v1}
MODELS=${MODELS:-roberta,bge_m3}
VARIANTS=${VARIANTS:-short,masked_short}
CLASSIFICATION_CONCURRENCY=${CLASSIFICATION_CONCURRENCY:-16}
EXPECTED_ROWS=${EXPECTED_ROWS:-$(${PYTHON} - "${PANEL}" <<'PY'
import sys
import pandas as pd
print(len(pd.read_parquet(sys.argv[1], columns=["row_index"])))
PY
)}

mkdir -p configs/generated logs/slurm_cninfo_all_pool_cls_launcher
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}
[[ -s "${PANEL}" ]] || { echo "label panel is missing: ${PANEL}" >&2; exit 2; }
[[ -d "${EMBEDDING_ROOT}" ]] || { echo "embedding root is missing: ${EMBEDDING_ROOT}" >&2; exit 2; }

manifest=${MANIFEST:-configs/generated/cninfo_2010_2026_short_all_pooling_seed42.tsv}
"${PYTHON}" scripts/build_pooled_classification_manifest.py \
  --phase all_pooling \
  --models "${MODELS}" \
  --variants "${VARIANTS}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --expected-rows "${EXPECTED_ROWS}" \
  --output "${manifest}"
tasks=$(($(wc -l < "${manifest}") - 1))
if (( tasks < 1 )); then
  echo "No pending all-pooling classification tasks" >&2
  exit 2
fi

tracked_sbatch() {
  "${PYTHON}" scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}
job_id=$(tracked_sbatch --parsable \
  --array="0-$((tasks - 1))%${CLASSIFICATION_CONCURRENCY}" \
  --export="ALL,MANIFEST=${manifest},PANEL=${PANEL},EXPECTED_ROWS=${EXPECTED_ROWS}" \
  scripts/slurm_pooled_embedding_classification.sbatch)
echo "submitted all-pooling classification tasks=${tasks} job=${job_id} rows=${EXPECTED_ROWS}"
