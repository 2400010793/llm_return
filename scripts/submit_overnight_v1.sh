#!/bin/bash
set -euo pipefail

: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
GPU_DEPENDENCY=${GPU_DEPENDENCY:-afterany:3974888:3974889}

pooled_tasks=$(( $(wc -l < configs/generated/overnight_pooled_v1.tsv) - 1 ))
dynamic_a_tasks=$(( $(wc -l < configs/generated/overnight_dynamic_v1_a.tsv) - 1 ))
dynamic_b_tasks=$(( $(wc -l < configs/generated/overnight_dynamic_v1_b.tsv) - 1 ))

MANIFEST=configs/generated/overnight_pooled_v1.tsv \
  sbatch --parsable --array="0-$((pooled_tasks - 1))%12" \
  --export=ALL,MANIFEST=configs/generated/overnight_pooled_v1.tsv \
  scripts/slurm_overnight_pooled_v1.sbatch

sbatch --parsable --dependency="${GPU_DEPENDENCY}" \
  --array="0-$((dynamic_a_tasks - 1))%1" \
  --export=ALL,MANIFEST=configs/generated/overnight_dynamic_v1_a.tsv,CUDA_DEVICE=5 \
  scripts/slurm_overnight_dynamic_v1.sbatch

sbatch --parsable --dependency="${GPU_DEPENDENCY}" \
  --array="0-$((dynamic_b_tasks - 1))%1" \
  --export=ALL,MANIFEST=configs/generated/overnight_dynamic_v1_b.tsv,CUDA_DEVICE=7 \
  scripts/slurm_overnight_dynamic_v1.sbatch
