#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return
MODE=${1:-fixed}
ROOT=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1
MANIFEST="$ROOT/results_all_embedding_cluster_regression/correct_manifest_${MODE}.tsv"
.venv/bin/python scripts/build_correct_embedding_cluster_manifest.py --mode "$MODE" --output "$MANIFEST"
N=$(( $(wc -l < "$MANIFEST") - 1 ))
MAX_ARRAY=1000
export CORRECT_EMB_MANIFEST="$MANIFEST"
if [ "$N" -le "$MAX_ARRAY" ]; then
  sbatch --array="0-$((N-1))%64" --export=CORRECT_EMB_MANIFEST="$MANIFEST",ARRAY_OFFSET=0 scripts/slurm_correct_embedding_cluster_regression.sbatch
else
  sbatch --array="0-$((MAX_ARRAY-1))%64" --export=CORRECT_EMB_MANIFEST="$MANIFEST",ARRAY_OFFSET=0 scripts/slurm_correct_embedding_cluster_regression.sbatch
  REMAINING=$((N-MAX_ARRAY))
  sbatch --array="0-$((REMAINING-1))%64" --export=CORRECT_EMB_MANIFEST="$MANIFEST",ARRAY_OFFSET="$MAX_ARRAY" scripts/slurm_correct_embedding_cluster_regression.sbatch
fi
echo "submitted $N $MODE folds"
