#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return

DATA_ROOT=${DATA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
PCA_ROOT=${PCA_ROOT:-${DATA_ROOT}/prompt_positive_v1/three_model_four_prompt_fair_pca_v2}
REPORT=${REPORT:-${DATA_ROOT}/prompt_positive_v1/REPORT_ALL_RESULTS.md}
SINA_PANEL=${SINA_PANEL:-${DATA_ROOT}/classification/sina_single_stock_classification_panel.parquet}
CNINFO_PANEL=${CNINFO_PANEL:-/mnt/lustre3/home/gaozh/llm_return/data/processed/cninfo_full_o2o_panel_2010_2026_hfq.parquet}
CPU_EXCLUDE=${CPU_EXCLUDE:-c001-epyc9755,c003-epyc9755,c005-epyc9755,c006-epyc9755,c010-epyc9755,c011-epyc9755,c012-epyc9755,c118-epyc9575f,v123-epyc9575f,v126-epyc9575f,v127-epyc9575f,v128-epyc9575f,v129-epyc9575f,v130-epyc9575f,v131-epyc9575f,v132-epyc9575f,v133-epyc9575f,v134-epyc9575f,v135-epyc9575f,v136-epyc9575f,v139-epyc9575f}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name three-model-four-prompt-fair-pca-v2 \
    --purpose "Fixed PCA32 comparison on Sina 6+2+1 and CNINFO 3+1; no raw embedding regression or per-fold hyperparameter search" \
    --input "${SINA_PANEL}" --input "${CNINFO_PANEL}" \
    --input "${DATA_ROOT}/prompt_positive_v1/embeddings" \
    --input "${DATA_ROOT}/qwen3_gguf_prompt_mean" \
    --output "${PCA_ROOT}" --output "${REPORT}" \
    --related-file scripts/build_three_model_four_prompt_fair.py \
    --related-file scripts/analyze_three_model_prompt_geometry.py \
    --related-file scripts/run_three_model_prompt_fair_fold.py \
    --related-file scripts/run_three_model_prompt_fusion.py \
    --related-file scripts/summarize_three_model_prompt_fair.py \
    --related-file scripts/slurm_three_model_prompt_build.sbatch \
    --related-file scripts/slurm_three_model_prompt_geometry.sbatch \
    --related-file scripts/slurm_three_model_prompt_rolling.sbatch \
    --related-file scripts/slurm_three_model_prompt_fusion.sbatch \
    --related-file scripts/slurm_three_model_prompt_summary.sbatch \
    --tag sina --tag cninfo --tag roberta --tag bge-m3 --tag qwen \
    --tag pca32 --tag fixed-parameters --tag token-body --tag clustering --tag fusion \
    --seed 42 \
    --snapshot-from-task 20260826T011835Z-qwen-cninfo-strict-token-full-v1-2505748 \
    -- bash "$0" "$@"
fi

mkdir -p "${PCA_ROOT}/sina" "${PCA_ROOT}/cninfo" logs/three_model_prompt_fair
tracked() {
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- "$@"
}

common_encoder=${DATA_ROOT}/prompt_positive_v1/embeddings
common_qwen=${DATA_ROOT}/qwen3_gguf_prompt_mean
sina_exports="ALL,OUTPUT_ROOT=${PCA_ROOT}/sina,PANEL=${SINA_PANEL},ENCODER_ROOT=${common_encoder}/sina,QWEN_ROOT=${common_qwen}/sina,DATASET=sina,ENCODER_SHARDS=64,ROW_INDEX_OFFSET=0,EXPECTED_ROWS=35576,HISTORY_YEARS=8,VALIDATION_YEARS=2,FIRST_TEST_YEAR=2018,REPORT=${REPORT}"
cninfo_exports="ALL,OUTPUT_ROOT=${PCA_ROOT}/cninfo,PANEL=${CNINFO_PANEL},ENCODER_ROOT=${common_encoder}/cninfo,QWEN_ROOT=${common_qwen}/cninfo,DATASET=cninfo,ENCODER_SHARDS=1000,ROW_INDEX_OFFSET=1,EXPECTED_ROWS=254544,HISTORY_YEARS=3,VALIDATION_YEARS=1,FIRST_TEST_YEAR=2021,REPORT=${REPORT}"

sina_build=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --export="${sina_exports}" scripts/slurm_three_model_prompt_build.sbatch)
cninfo_build=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --export="${cninfo_exports}" scripts/slurm_three_model_prompt_build.sbatch)
sina_geometry=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${sina_build}" --export="${sina_exports}" scripts/slurm_three_model_prompt_geometry.sbatch)
cninfo_geometry=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${cninfo_build}" --export="${cninfo_exports}" scripts/slurm_three_model_prompt_geometry.sbatch)
sina_rolling=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${sina_build}" --array=0-215%24 --export="${sina_exports}" scripts/slurm_three_model_prompt_rolling.sbatch)
cninfo_rolling=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${cninfo_build}" --array=0-143%24 --export="${cninfo_exports}" scripts/slurm_three_model_prompt_rolling.sbatch)
sina_fusion=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${sina_rolling}" --array=0-8%9 --export="${sina_exports}" scripts/slurm_three_model_prompt_fusion.sbatch)
cninfo_fusion=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${cninfo_rolling}" --array=0-5%6 --export="${cninfo_exports}" scripts/slurm_three_model_prompt_fusion.sbatch)
sina_summary=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${sina_geometry}:${sina_fusion}" --export="${sina_exports}" scripts/slurm_three_model_prompt_summary.sbatch)
cninfo_summary=$(tracked sbatch --parsable --exclude="${CPU_EXCLUDE}" --dependency="afterok:${cninfo_geometry}:${cninfo_fusion}:${sina_summary}" --export="${cninfo_exports}" scripts/slurm_three_model_prompt_summary.sbatch)

echo "sina: build=${sina_build} geometry=${sina_geometry} rolling=${sina_rolling} fusion=${sina_fusion} summary=${sina_summary}"
echo "cninfo: build=${cninfo_build} geometry=${cninfo_geometry} rolling=${cninfo_rolling} fusion=${cninfo_fusion} summary=${cninfo_summary}"
