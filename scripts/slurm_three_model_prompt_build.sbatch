#!/bin/bash
#SBATCH --job-name=three-model-build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/three_model_prompt_fair/%j_build.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/three_model_prompt_fair/%j_build.err
set -euo pipefail
: "${OUTPUT_ROOT:?}" "${PANEL:?}" "${ENCODER_ROOT:?}" "${QWEN_ROOT:?}"
: "${DATASET:?}" "${ENCODER_SHARDS:?}" "${ROW_INDEX_OFFSET:?}" "${EXPECTED_ROWS:?}" "${HISTORY_YEARS:?}" "${VALIDATION_YEARS:?}"
cd /data/alpha_team2/shares/llm_return
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/build_three_model_four_prompt_fair.py \
  --panel "${PANEL}" --encoder-root "${ENCODER_ROOT}" --qwen-root "${QWEN_ROOT}" \
  --output-root "${OUTPUT_ROOT}" --dataset "${DATASET}" \
  --encoder-shards "${ENCODER_SHARDS}" --row-index-offset "${ROW_INDEX_OFFSET}" \
  --expected-rows "${EXPECTED_ROWS}" --history-years "${HISTORY_YEARS}" \
  --validation-years "${VALIDATION_YEARS}"
