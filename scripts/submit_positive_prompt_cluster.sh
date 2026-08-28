#!/bin/bash
set -euo pipefail
ROOT=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1; REPO=/data/alpha_team2/shares/llm_return
MANIFEST=$ROOT/cluster_manifest_sina.tsv; .venv/bin/python scripts/build_positive_prompt_matrix_manifest.py --output "$MANIFEST" --dataset sina --shards 64
sbatch --parsable --array=0-63%16 --export="ALL,MANIFEST=$MANIFEST" "$REPO/scripts/slurm_positive_prompt_cluster.sbatch"
