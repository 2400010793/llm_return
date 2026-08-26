#!/bin/bash
#SBATCH --job-name=sina-cninfo-launch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_sina_cninfo_method/%j_launch.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_sina_cninfo_method/%j_launch.err

set -euo pipefail
cd /home/gaozh/llm_return
: "${TASK_RECORD_ID:?launcher must be tracked}"
classification_manifest=configs/generated/sina_cninfo_method_classification_v1.tsv
regression_manifest=configs/generated/sina_cninfo_method_regression_v1.tsv
.venv/bin/python scripts/build_sina_cninfo_method_downstream_manifests.py \
  --classification-manifest "${classification_manifest}" \
  --regression-manifest "${regression_manifest}"
classification_tasks=$(($(wc -l < "${classification_manifest}") - 1))
regression_tasks=$(($(wc -l < "${regression_manifest}") - 1))
tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}
classification_job=$(tracked_sbatch --parsable --array="0-$((classification_tasks - 1))%4" \
  --export="ALL,MANIFEST=${classification_manifest}" scripts/slurm_sina_cninfo_method_classification.sbatch)
regression_job=$(tracked_sbatch --parsable --array="0-$((regression_tasks - 1))%4" \
  --export="ALL,MANIFEST=${regression_manifest}" scripts/slurm_sina_cninfo_method_regression.sbatch)
echo "classification=${classification_job} tasks=${classification_tasks} concurrency=4"
echo "regression=${regression_job} tasks=${regression_tasks} concurrency=4"
