#!/bin/bash
# One variant x (ungated + Fisher/variance keep 1/2/4/8).
#SBATCH --job-name=prompt-token-clean-pca
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --array=0-8%4
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_prompt_token_cls_clean/pca_%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_prompt_token_cls_clean/pca_%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MODEL:?export MODEL=roberta or bge_m3}"
: "${VARIANT:?export VARIANT=short or masked_short}"
: "${SLURM_ARRAY_TASK_ID:?run as a Slurm array}"

PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
INPUT_ROOT=${INPUT_ROOT:-data/processed/prompt_token_embeddings_2010_2026_clean_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/processed/prompt_token_pca_2010_2026_clean_v1}
keeps=(1 2 4 8)
variant=${VARIANT}
config=${SLURM_ARRAY_TASK_ID}
gate=none
keep=8
if (( config >= 1 && config <= 4 )); then
  gate=fisher
  keep=${keeps[$((config - 1))]}
elif (( config >= 5 )); then
  gate=variance
  keep=${keeps[$((config - 5))]}
fi

output_dir=${OUTPUT_ROOT}/${MODEL}/${variant}
mkdir -p logs/slurm_prompt_token_cls_clean "${output_dir}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

.venv/bin/python scripts/run_prompt_token_pca_v5.py "${PANEL}" \
  --input-root "${INPUT_ROOT}" --output-dir "${output_dir}" \
  --model "${MODEL}" --variant "${variant}" \
  --components 128 --fit-sample-rows 10000 --transform-batch-size 256 \
  --token-gate-method "${gate}" --token-gate-keep "${keep}" \
  --target-column next_day_return --expected-shards 64 --expected-rows 903665 \
  --require-completed --seed 42
