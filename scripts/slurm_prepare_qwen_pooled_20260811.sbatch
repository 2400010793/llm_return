#!/bin/bash
#SBATCH --job-name=qwen-pool-prepare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/home/team/llm_return/logs/slurm_qwen_overnight_20260811/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_qwen_overnight_20260811/%j.err

set -euo pipefail
REPO_ROOT=/home/team/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
mkdir -p logs/slurm_qwen_overnight_20260811 \
  data/processed/pooled_embeddings_qwen_candidates_1000
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${REPO_ROOT}/.venv/bin/python" scripts/prepare_qwen_pooled_candidates.py \
  data/processed/embeddings/qwen3_embedding_8b_candidates_1000 \
  data/processed/pooled_embeddings_qwen_candidates_1000 \
  --views short,masked_short,plain \
  --expected-parts 71 --expected-rows 350577 --expected-dimension 4096
