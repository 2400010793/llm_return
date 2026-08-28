#!/bin/bash
set -euo pipefail
ROOT=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1
REPO=/data/alpha_team2/shares/llm_return
DATASET=${DATASET:-sina}
SHARDS=${SHARDS:-64}
MANIFEST=${MANIFEST:-$ROOT/matrix_manifest_${DATASET}.tsv}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/matrices}
mkdir -p "$REPO/logs/positive_prompt_matrix" "$OUTPUT_ROOT"
.venv/bin/python "$REPO/scripts/build_positive_prompt_matrix_manifest.py" \
  --output "$MANIFEST" --dataset "$DATASET" --shards "$SHARDS"
tasks=$(($(wc -l < "$MANIFEST") - 1))
sbatch --parsable --array="0-$((tasks-1))%32" \
  --export="ALL,MANIFEST=$MANIFEST,EMBEDDINGS_ROOT=$ROOT/embeddings,OUTPUT_ROOT=$OUTPUT_ROOT" \
  "$REPO/scripts/slurm_build_positive_prompt_matrix.sbatch"
