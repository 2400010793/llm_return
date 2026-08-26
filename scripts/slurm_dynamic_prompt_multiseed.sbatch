#!/bin/bash
#SBATCH --job-name=dynamic-prompt-ms
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%1
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_dynamic_prompt/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_dynamic_prompt/%A_%a.err

set -euo pipefail
REPO_ROOT=/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?dynamic prompt multiseed must use scripts/task_tracker.py}"
: "${MATRIX_GROUP:?submit with MATRIX_GROUP=A or B}"
: "${SLURM_ARRAY_TASK_ID:?run this entrypoint as a Slurm array}"

case "${MATRIX_GROUP}" in
  A)
    CUDA_DEVICE=5
    COMBINATIONS=(
      "13 uniform" "13 dynamic"
      "2024 uniform" "2024 dynamic"
      "2025 uniform" "2025 dynamic"
    )
    ;;
  B)
    CUDA_DEVICE=7
    COMBINATIONS=(
      "13 static" "2024 static" "2025 static" "2026 static"
      "2026 uniform" "2026 dynamic"
    )
    ;;
  *) echo "invalid MATRIX_GROUP=${MATRIX_GROUP}" >&2; exit 2 ;;
esac

read -r SEED GATE_MODE <<< "${COMBINATIONS[SLURM_ARRAY_TASK_ID]}"
MAX_EPOCHS=${MAX_EPOCHS:-50}
PATIENCE=${PATIENCE:-7}
OUTPUT="reports/dynamic_prompt/multiseed/roberta_masked_short_${GATE_MODE}_regression_seed${SEED}.json"

mkdir -p logs/slurm_dynamic_prompt reports/dynamic_prompt/multiseed
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

