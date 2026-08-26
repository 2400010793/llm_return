#!/bin/bash
set -euo pipefail

: "${TASK_RECORD_ID:?multiseed submission must use scripts/task_tracker.py}"
MATRIX_GROUP=A sbatch --parsable scripts/slurm_dynamic_prompt_multiseed.sbatch
MATRIX_GROUP=B sbatch --parsable scripts/slurm_dynamic_prompt_multiseed.sbatch
