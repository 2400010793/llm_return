#!/bin/bash
#SBATCH --job-name=qwen-linear-select
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_qwen_linear/%j_select.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_qwen_linear/%j_select.err

set -euo pipefail
cd /data/alpha_team2/shares/llm_return
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${SCREEN_MANIFEST:?export SCREEN_MANIFEST}"
ROOT=reports/regression/qwen_2018_2026_linear_v1
STRICT=configs/generated/qwen_linear_strict_6_2_1.tsv
ROLLING=configs/generated/qwen_linear_rolling_3_1_1.tsv
.venv/bin/python scripts/select_qwen_linear_candidates.py \
  --screen-manifest "${SCREEN_MANIFEST}" --output-root "${ROOT}" \
  --strict-manifest "${STRICT}" --rolling-manifest "${ROLLING}"

strict_tasks=$(( $(wc -l < "${STRICT}") - 1 ))
rolling_tasks=$(( $(wc -l < "${ROLLING}") - 1 ))
strict_id=$(.venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
  sbatch --parsable --array="0-$((strict_tasks - 1))%3" \
  --export=ALL,MANIFEST="${STRICT}",FIT_WINDOW_YEARS=6,VALIDATION_WINDOW_YEARS=2 \
  scripts/slurm_qwen_linear_regression.sbatch)
rolling_id=$(.venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
  sbatch --parsable --array="0-$((rolling_tasks - 1))%3" \
  --export=ALL,MANIFEST="${ROLLING}",FIT_WINDOW_YEARS=3,VALIDATION_WINDOW_YEARS=1 \
  scripts/slurm_qwen_linear_regression.sbatch)
cluster_strict_id=$(.venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
  sbatch --parsable --export=ALL,FIT_WINDOW_YEARS=6,VALIDATION_WINDOW_YEARS=2,OUTPUT_ROOT="${ROOT}/cluster_strict_6_2_1" \
  scripts/slurm_qwen_cluster_linear.sbatch)
cluster_rolling_id=$(.venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
  sbatch --parsable --export=ALL,FIT_WINDOW_YEARS=3,VALIDATION_WINDOW_YEARS=1,OUTPUT_ROOT="${ROOT}/cluster_rolling_3_1_1" \
  scripts/slurm_qwen_cluster_linear.sbatch)
final_id=$(.venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
  sbatch --parsable --dependency=afterok:${strict_id}:${rolling_id}:${cluster_strict_id}:${cluster_rolling_id} \
  --export=ALL,STRICT_MANIFEST="${STRICT}",ROLLING_MANIFEST="${ROLLING}" \
  scripts/slurm_qwen_linear_finalize.sbatch)
echo "strict=${strict_id} rolling=${rolling_id} cluster_strict=${cluster_strict_id} cluster_rolling=${cluster_rolling_id} final=${final_id}"
