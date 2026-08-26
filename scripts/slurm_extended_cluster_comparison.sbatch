#!/bin/bash
# Run one PCA/KMeans Token-vs-Body cluster comparison on an existing matrix.
#SBATCH --job-name=extended-factor-cluster
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/extended_factor_cluster/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/extended_factor_cluster/%A_%a.err
set -euo pipefail
: "${MANIFEST:?MANIFEST is required}"
REPO=/mnt/lustre3/home/gaozh/llm_return
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
IFS=$'\t' read -r axis representation target panel matrix metadata output <<< "${line%$'\r'}"
mkdir -p "$REPO/logs/extended_factor_cluster" "$(dirname "$output")"
cd "$REPO"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/run_aligned_factor_cluster_comparison.py \
  --panel "$panel" --matrix "$matrix" --metadata "$metadata" \
  --output "$output" --axis "$axis" --representation "$representation" \
  --target "$target" --first-year 2026 --last-year 2026 \
  --sample-rows 20000 --skip-umap
