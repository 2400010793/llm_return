#!/bin/bash
#SBATCH --job-name=prompt-pca-fuse
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_prompt_pca_fusion/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_prompt_pca_fusion/%A_%a.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"

PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-data/processed/pooled_embeddings_2010_2026_roberta_v1}
PCA_ROOT=${PCA_ROOT:-data/processed/prompt_token_pca_2010_2026_clean_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/processed/prompt_pca_segment_features_2010_2026_clean_v1}
EXPECTED_ROWS=${EXPECTED_ROWS:-903665}
variants=(short masked_short)
variant=${variants[${SLURM_ARRAY_TASK_ID}]}

mkdir -p logs/slurm_prompt_pca_fusion
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 scripts/compose_prompt_pca_segment_features.py "${PANEL}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --prompt-pca-dir "${PCA_ROOT}/roberta/${variant}" \
  --output-dir "${OUTPUT_ROOT}/roberta/${variant}" \
  --model roberta \
  --variant "${variant}" \
  --components 128 \
  --expected-rows "${EXPECTED_ROWS}"
