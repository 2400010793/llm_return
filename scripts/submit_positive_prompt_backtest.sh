#!/bin/bash
set -euo pipefail
ROOT=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1
REPO=/data/alpha_team2/shares/llm_return
MANIFEST=${MANIFEST:-$ROOT/summary/simple_states_manifest.tsv}
tasks=$(($(wc -l < "$MANIFEST") - 1))
sbatch --parsable --array="0-$((tasks-1))%16" --export="ALL,MANIFEST=$MANIFEST" "$REPO/scripts/slurm_positive_prompt_simple_states.sbatch"
