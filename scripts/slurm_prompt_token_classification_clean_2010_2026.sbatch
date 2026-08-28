#!/bin/bash
# One variant x 12 rolling classification cells.
#SBATCH --job-name=prompt-token-clean-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-11%8
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_prompt_token_cls_clean/cls_%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_prompt_token_cls_clean/cls_%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MODEL:?export MODEL=roberta or bge_m3}"
: "${VARIANT:?export VARIANT=short or masked_short}"
: "${SLURM_ARRAY_TASK_ID:?run as a Slurm array}"

PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
PCA_ROOT=${PCA_ROOT:-data/processed/prompt_token_pca_2010_2026_clean_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/classification/prompt_tokens_2010_2026_clean_v1}
keeps=(1 2 4 8)
base_classifiers=(logistic sgd simple_mlp mlp)
variant=${VARIANT}
config=${SLURM_ARRAY_TASK_ID}
gate=none
keep=0
classifier=logistic
if (( config < 4 )); then
  classifier=${base_classifiers[${config}]}
elif (( config < 8 )); then
  gate=fisher
  keep=${keeps[$((config - 4))]}
else
  gate=variance
  keep=${keeps[$((config - 8))]}
fi

base=${PCA_ROOT}/${MODEL}/${variant}
if [[ "${gate}" == none ]]; then
  matrix=${base}/pca_128.npy
  summary=${base}/summary.json
  tag=ordered_flat_pca128_${classifier}
else
  matrix=${base}/${gate}_gate${keep}_pca_128.npy
  summary=${base}/${gate}_gate${keep}_pca_128.summary.json
  tag=${gate}_gate${keep}_pca128_logistic
fi
output=${OUTPUT_ROOT}/${MODEL}_${variant}_${tag}.json
mkdir -p logs/slurm_prompt_token_cls_clean "${OUTPUT_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

.venv/bin/python scripts/run_reduced_prompt_classification.py "${PANEL}" \
  --matrix "${matrix}" --matrix-summary "${summary}" \
  --model "${MODEL}" --variant "${variant}" --classifier "${classifier}" \
  --components 128 --train-target-column next_day_return \
  --evaluation-target-column next_day_return --seed 42 --output "${output}"
