#!/bin/bash
#SBATCH --job-name=cninfo-tree-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/cninfo_tree_stacking/%j.report.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/cninfo_tree_stacking/%j.report.err

set -euo pipefail
cd /data/alpha_team2/shares/llm_return
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${RESULT_DIR:?export RESULT_DIR}"
: "${MAIN_REPORT:?export MAIN_REPORT}"
.venv/bin/python scripts/update_report_cninfo_tree_stacking.py \
  --result-dir "${RESULT_DIR}" --report "${MAIN_REPORT}"
