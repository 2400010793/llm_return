#!/bin/bash
# Classify corrected v5 Prompt-token representations after hard gating and PCA.
# 2 models x 4 variants x 2 gates x 4 classifiers. Submit only after gate-PCA audit.
#SBATCH --job-name=prompt-gate-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-63%8
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_prompt_gate_cls_v5/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_prompt_gate_cls_v5/%A_%a.err
set -euo pipefail
cd /home/gaozh/llm_return
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
mkdir -p logs/slurm_prompt_gate_cls_v5 reports/classification/prompt_tokens_v5_gated
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
models=(roberta bge_m3)
variants=(short masked_short long masked_long)
gates=(fisher variance)
classifiers=(logistic linear_svm sgd mlp)
model=${models[$((SLURM_ARRAY_TASK_ID / 32))]}
variant=${variants[$(((SLURM_ARRAY_TASK_ID / 8) % 4))]}
gate=${gates[$(((SLURM_ARRAY_TASK_ID / 4) % 2))]}
classifier=${classifiers[$((SLURM_ARRAY_TASK_ID % 4))]}
base="data/processed/prompt_token_pca_v5/${model}/${variant}/${gate}_gate8_pca_128"
output="reports/classification/prompt_tokens_v5_gated/${model}_${variant}_${gate}_gate8_${classifier}_pca128.json"
/usr/bin/python3 scripts/run_reduced_prompt_classification.py \
  data/processed/cninfo_full_classification_panel.parquet \
  --matrix "${base}.npy" --matrix-summary "${base}.summary.json" \
  --model "${model}" --variant "${variant}" --classifier "${classifier}" \
  --components 128 --seed 42 --output "${output}"
