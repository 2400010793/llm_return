#!/bin/bash
#SBATCH --job-name=o2o-filter-reg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_paper_filters/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_paper_filters/%A_%a.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?regression array must be tracked}"
: "${MANIFEST:?export MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id panel filter_id output <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || { echo "manifest task mismatch" >&2; exit 2; }
[[ -s "${panel}" ]] || { echo "missing filtered panel: ${panel}" >&2; exit 2; }

/usr/bin/python3 scripts/run_pooled_embedding_regression.py "${panel}" \
  --embedding-root data/processed/pooled_embeddings_v3 \
  --model roberta --variant masked_short --feature title_body_full_concat \
  --target-column next_day_open_to_open_winsor_residual \
  --regressor huber_sgd --run-mode screen --search-stage fine \
  --reducer pca --reducer-components 256 \
  --expected-embedding-rows 350577 --seed 42 --output "${output}"
