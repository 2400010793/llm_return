#!/bin/bash
#SBATCH --job-name=cninfo-progress-monitor
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=1-01:00:00
#SBATCH --array=0-6%1
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_pipeline_monitor/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_pipeline_monitor/%A_%a.err

set -u
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?pipeline monitor must use scripts/task_tracker.py}"
mkdir -p logs/slurm_pipeline_monitor reports

for _snapshot in $(seq 1 48); do
  "${REPO_ROOT}/.venv/bin/python" scripts/save_pipeline_progress_snapshot.py \
    --output reports/cninfo_pipeline_progress_30min.jsonl \
    --latest reports/cninfo_pipeline_progress_latest.json
  sleep 1800
done
