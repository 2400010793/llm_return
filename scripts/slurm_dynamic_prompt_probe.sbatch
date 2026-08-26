#!/bin/bash
#SBATCH --job-name=dynamic-prompt-probe
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_dynamic_prompt/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_dynamic_prompt/%j.err

set -euo pipefail
REPO_ROOT=/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?dynamic prompt probe must be submitted through scripts/task_tracker.py}"

TASK=${TASK:-regression}
GATE_MODE=${GATE_MODE:-dynamic}
MAX_EPOCHS=${MAX_EPOCHS:-3}
PATIENCE=${PATIENCE:-2}
SEED=${SEED:-42}
# Device 9 on c018-10gpu is visible to nvidia-smi but currently fails CUDA
# initialization. Device 7 is the default probe device; callers may override it.
CUDA_DEVICE=${CUDA_DEVICE:-7}
OUTPUT=${OUTPUT:-reports/dynamic_prompt/probes/roberta_masked_short_${GATE_MODE}_${TASK}_seed${SEED}.json}

mkdir -p logs/slurm_dynamic_prompt reports/dynamic_prompt/probes
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
"${REPO_ROOT}/.venv/bin/python" scripts/run_dynamic_prompt_token_gating.py \
  data/processed/cninfo_full_classification_panel.parquet \
  --input-root data/processed/prompt_token_embeddings_v5 \
  --model roberta --variant masked_short \
  --task "${TASK}" --gate-mode "${GATE_MODE}" --run-mode screen \
  --batch-size 256 --prediction-batch-size 512 \
  --max-epochs "${MAX_EPOCHS}" --patience "${PATIENCE}" \
  --seed "${SEED}" --device cuda --output "${OUTPUT}"
