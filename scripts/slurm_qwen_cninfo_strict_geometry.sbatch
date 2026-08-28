#!/bin/bash
#SBATCH --job-name=qwen-cninfo-geometry
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/qwen_cninfo_strict/%j.geometry.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/qwen_cninfo_strict/%j.geometry.err

set -euo pipefail
cd /data/alpha_team2/shares/llm_return
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${INTERSECTION_ROOT:?export INTERSECTION_ROOT}"
: "${MATRIX_ROOT:?export MATRIX_ROOT}"
: "${GEOMETRY_ROOT:?export GEOMETRY_ROOT}"
mkdir -p "${GEOMETRY_ROOT}" logs/qwen_cninfo_strict

.venv/bin/python scripts/analyze_qwen_cninfo_token_full_geometry.py \
  --matrix-root "${MATRIX_ROOT}" \
  --panel "${INTERSECTION_ROOT}/intersection_panel.parquet" \
  --target-column next_day_open_to_open_return \
  --output-dir "${GEOMETRY_ROOT}"
