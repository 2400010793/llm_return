#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

QWEN_ROOT=${QWEN_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/qwen3_gguf_prompt_mean/cninfo}
PANEL=${PANEL:-/data/alpha_team2/shares/llm_return/data/processed/cninfo_full_o2o_panel_2010_2026_hfq.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/qwen_cninfo_strict_min_intersection_v1}
INTERSECTION_ROOT=${INTERSECTION_ROOT:-${OUTPUT_ROOT}/intersection}
MATRIX_ROOT=${MATRIX_ROOT:-${OUTPUT_ROOT}/matrices}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-${OUTPUT_ROOT}/geometry}
REGRESSION_ROOT=${REGRESSION_ROOT:-${OUTPUT_ROOT}/regression}
SUMMARY_ROOT=${SUMMARY_ROOT:-${OUTPUT_ROOT}/summary}
CPU_EXCLUDE=${CPU_EXCLUDE:-c001-epyc9755,c003-epyc9755,c005-epyc9755,c006-epyc9755,c010-epyc9755,c011-epyc9755,c012-epyc9755,c118-epyc9575f,v123-epyc9575f,v126-epyc9575f,v127-epyc9575f,v128-epyc9575f,v129-epyc9575f,v130-epyc9575f,v131-epyc9575f,v132-epyc9575f,v133-epyc9575f,v134-epyc9575f,v135-epyc9575f,v136-epyc9575f,v139-epyc9575f}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name qwen-cninfo-strict-token-full-v1 \
    --purpose "Strict CNINFO Qwen target-token versus article representation geometry, PCA-Ridge OOS prediction, factor correlation, and Top20 overlap" \
    --input "${QWEN_ROOT}" --input "${PANEL}" --output "${OUTPUT_ROOT}" \
    --related-file scripts/prepare_qwen_cninfo_strict_intersection.py \
    --related-file scripts/build_qwen_cninfo_strict_matrix.py \
    --related-file scripts/analyze_qwen_cninfo_token_full_geometry.py \
    --related-file scripts/summarize_qwen_cninfo_strict_comparison.py \
    --related-file scripts/run_sina_precomputed_regression.py \
    --related-file scripts/slurm_qwen_cninfo_strict_matrix.sbatch \
    --related-file scripts/slurm_qwen_cninfo_strict_geometry.sbatch \
    --related-file scripts/slurm_qwen_cninfo_strict_regression.sbatch \
    --related-file scripts/slurm_qwen_cninfo_strict_summary.sbatch \
    --tag qwen --tag cninfo --tag token-embedding --tag strict-621 --tag factor-correlation \
    --seed 42 -- bash "$0" "$@"
fi

mkdir -p "${INTERSECTION_ROOT}" "${MATRIX_ROOT}" "${GEOMETRY_ROOT}" \
  "${REGRESSION_ROOT}" "${SUMMARY_ROOT}" logs/qwen_cninfo_strict
.venv/bin/python scripts/prepare_qwen_cninfo_strict_intersection.py \
  --root "${QWEN_ROOT}" --panel "${PANEL}" --output-dir "${INTERSECTION_ROOT}"

exports="ALL,QWEN_ROOT=${QWEN_ROOT},INTERSECTION_ROOT=${INTERSECTION_ROOT},MATRIX_ROOT=${MATRIX_ROOT},GEOMETRY_ROOT=${GEOMETRY_ROOT},REGRESSION_ROOT=${REGRESSION_ROOT},SUMMARY_ROOT=${SUMMARY_ROOT}"
matrix_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --exclude="${CPU_EXCLUDE}" --array=0-9%10 \
      --export="${exports}" scripts/slurm_qwen_cninfo_strict_matrix.sbatch
)
geometry_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${matrix_job}" \
      --export="${exports}" scripts/slurm_qwen_cninfo_strict_geometry.sbatch
)
regression_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --exclude="${CPU_EXCLUDE}" --array=0-9%5 \
      --dependency="afterok:${matrix_job}" --export="${exports}" \
      scripts/slurm_qwen_cninfo_strict_regression.sbatch
)
summary_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --exclude="${CPU_EXCLUDE}" \
      --dependency="afterok:${geometry_job}:${regression_job}" --export="${exports}" \
      scripts/slurm_qwen_cninfo_strict_summary.sbatch
)

echo "matrix=${matrix_job} geometry=${geometry_job} regression=${regression_job} summary=${summary_job}"
