#!/bin/bash
# Compare full soft gates and training-only hard keep-4 prompt gates on O2O.
# Tasks 0..5: classification/regression x uniform/static/dynamic on all tokens.
# Tasks 6..9: classification/regression x Fisher/variance hard keep-4 + uniform.
#SBATCH --job-name=prompt-o2o-gate
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-9%1
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_prompt_o2o_gate/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_prompt_o2o_gate/%A_%a.err

set -euo pipefail
REPO_ROOT=/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?run as a Slurm array over 0..9}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) TASK=classification; GATE=uniform; HARD=none ;;
  1) TASK=classification; GATE=static; HARD=none ;;
  2) TASK=classification; GATE=dynamic; HARD=none ;;
  3) TASK=regression; GATE=uniform; HARD=none ;;
  4) TASK=regression; GATE=static; HARD=none ;;
  5) TASK=regression; GATE=dynamic; HARD=none ;;
  6) TASK=classification; GATE=uniform; HARD=fisher ;;
  7) TASK=classification; GATE=uniform; HARD=variance ;;
  8) TASK=regression; GATE=uniform; HARD=fisher ;;
  9) TASK=regression; GATE=uniform; HARD=variance ;;
  *) echo "invalid task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

# GPU 1 was verified idle before submission.  Keep this matrix single-file so it
# does not collide with the long-running Ollama/Qwen services on the other GPUs.
CUDA_DEVICE=1
SEED=${SEED:-42}
MAX_EPOCHS=${MAX_EPOCHS:-50}
PATIENCE=${PATIENCE:-7}
HARD_TAG=""
HARD_ARGS=()
if [[ "${HARD}" != "none" ]]; then
  HARD_TAG="_hard-${HARD}4"
  HARD_ARGS=(--hard-gate-method "${HARD}" --hard-gate-keep 4)
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/dynamic_prompt/o2o_seed42}
TRAIN_DEVICE=${TRAIN_DEVICE:-cuda}
OUTPUT="${OUTPUT_ROOT}/roberta_masked_short_${GATE}${HARD_TAG}_${TASK}_seed${SEED}.json"

mkdir -p logs/slurm_prompt_o2o_gate "${OUTPUT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

"${REPO_ROOT}/.venv/bin/python" scripts/run_dynamic_prompt_token_gating.py \
  data/processed/cninfo_full_o2o_panel.parquet \
  --input-root data/processed/prompt_token_embeddings_v5 \
  --model roberta --variant masked_short \
  --task "${TASK}" --gate-mode "${GATE}" --run-mode screen \
  --return-column next_day_open_to_open_return \
  --classification-target-column next_day_open_to_open_label \
  --batch-size 256 --prediction-batch-size 512 \
  --max-epochs "${MAX_EPOCHS}" --patience "${PATIENCE}" \
  --seed "${SEED}" --device "${TRAIN_DEVICE}" --output "${OUTPUT}" \
  "${HARD_ARGS[@]}"
