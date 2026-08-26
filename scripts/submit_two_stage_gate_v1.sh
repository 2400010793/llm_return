#!/bin/bash
set -euo pipefail

: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
GPU_DEPENDENCY=${GPU_DEPENDENCY:-afterany:3974888:3974889}

gpu_a_tasks=$(( $(wc -l < configs/generated/two_stage_gate_v1_a.tsv) - 1 ))
gpu_b_tasks=$(( $(wc -l < configs/generated/two_stage_gate_v1_b.tsv) - 1 ))
predictor_tasks=$(( $(wc -l < configs/generated/two_stage_gate_v1_predictors.tsv) - 1 ))

gpu_a_job=$(sbatch --parsable --dependency="${GPU_DEPENDENCY}" \
  --array="0-$((gpu_a_tasks - 1))%1" \
  --export=ALL,MANIFEST=configs/generated/two_stage_gate_v1_a.tsv,CUDA_DEVICE=5 \
  scripts/slurm_frozen_gate_representation_v1.sbatch)
echo "${gpu_a_job}"

gpu_b_job=$(sbatch --parsable --dependency="${GPU_DEPENDENCY}" \
  --array="0-$((gpu_b_tasks - 1))%1" \
  --export=ALL,MANIFEST=configs/generated/two_stage_gate_v1_b.tsv,CUDA_DEVICE=7 \
  scripts/slurm_frozen_gate_representation_v1.sbatch)
echo "${gpu_b_job}"

predictor_job=$(sbatch --parsable --dependency="afterok:${gpu_a_job}:${gpu_b_job}" \
  --array="0-$((predictor_tasks - 1))%8" \
  --export=ALL,MANIFEST=configs/generated/two_stage_gate_v1_predictors.tsv \
  scripts/slurm_frozen_representation_predictors_v1.sbatch)
echo "${predictor_job}"

summary_job=$(sbatch --parsable --dependency="afterany:${predictor_job}" \
  scripts/slurm_summarize_two_stage_gate_v1.sbatch)
echo "${summary_job}"
