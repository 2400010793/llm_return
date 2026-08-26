#!/bin/bash
set -euo pipefail
ROOT=/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1
REPO=/mnt/lustre3/home/gaozh/llm_return
DATASET=${DATASET:-sina}; PANEL=${PANEL:-$ROOT/../classification/sina_single_stock_classification_panel.parquet}
MATRIX_ROOT=${MATRIX_ROOT:-$ROOT/matrices}; OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/results}; EXPECTED_ROWS=${EXPECTED_ROWS:-75894}; REPRESENTATIONS=${REPRESENTATIONS:-prompt_mean,return_span}
MANIFEST=${MANIFEST:-$ROOT/downstream_manifest_${DATASET}.tsv}
mkdir -p "$REPO/logs/positive_prompt_downstream" "$OUTPUT_ROOT"
.venv/bin/python "$REPO/scripts/build_positive_prompt_downstream_manifest.py" --output "$MANIFEST" --dataset "$DATASET" --panel "$PANEL" --matrix-root "$MATRIX_ROOT" --output-root "$OUTPUT_ROOT" --representations "$REPRESENTATIONS"
tasks=$(($(wc -l < "$MANIFEST") - 1))
sbatch --parsable --dependency="afterok:${MATRIX_JOB:?MATRIX_JOB required}" --array="0-$((tasks-1))%32" --export="ALL,MANIFEST=$MANIFEST,EXPECTED_ROWS=$EXPECTED_ROWS" "$REPO/scripts/slurm_positive_prompt_downstream.sbatch"
