#!/bin/bash
set -euo pipefail
ROOT=/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1
REPO=/mnt/lustre3/home/gaozh/llm_return
MANIFEST=${MANIFEST:-$ROOT/summary/simple_states_manifest.tsv}
tasks=$(($(wc -l < "$MANIFEST") - 1))
sbatch --parsable --array="0-$((tasks-1))%16" --export="ALL,MANIFEST=$MANIFEST" "$REPO/scripts/slurm_positive_prompt_simple_states.sbatch"
