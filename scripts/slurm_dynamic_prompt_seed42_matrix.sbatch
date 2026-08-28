#!/bin/bash
#SBATCH --job-name=dynamic-prompt-s42
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/home/team/llm_return/logs/slurm_dynamic_prompt/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_dynamic_prompt/%A_%a.err

set -euo pipefail
REPO_ROOT=/home/team/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?dynamic prompt matrix must be submitted through scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?run this entrypoint as a Slurm array}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) GATE_MODE=uniform ;;
  1) GATE_MODE=static ;;
  2) GATE_MODE=dynamic ;;
  *) echo "invalid task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

# The array runs at most two tasks. Alternating between two verified idle
# devices prevents tasks from selecting the same physical GPU.
CUDA_DEVICES=(5 7)
CUDA_DEVICE=${CUDA_DEVICES[$((SLURM_ARRAY_TASK_ID % 2))]}
MAX_EPOCHS=${MAX_EPOCHS:-50}
PATIENCE=${PATIENCE:-7}
SEED=${SEED:-42}
OUTPUT="reports/dynamic_prompt/seed42/roberta_masked_short_${GATE_MODE}_regression_seed${SEED}.json"

mkdir -p logs/slurm_dynamic_prompt reports/dynamic_prompt/seed42
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

"${REPO_ROOT}/.venv/bin/python" scripts/run_dynamic_prompt_token_gating.py \
  data/processed/cninfo_full_classification_panel.parquet \
  --input-root data/processed/prompt_token_embeddings_v5 \
  --model roberta --variant masked_short \
  --task regression --gate-mode "${GATE_MODE}" --run-mode screen \
  --batch-size 256 --prediction-batch-size 512 \
  --max-epochs "${MAX_EPOCHS}" --patience "${PATIENCE}" \
  --seed "${SEED}" --device cuda --output "${OUTPUT}"
