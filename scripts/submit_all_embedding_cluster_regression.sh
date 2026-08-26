#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
ROOT=/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1
MODE=${1:-fixed}
MANIFEST="$ROOT/results_all_embedding_cluster_regression/manifest_${MODE}.tsv"
.venv/bin/python scripts/build_all_embedding_cluster_manifest.py --mode "$MODE" --output "$MANIFEST"
N=$(( $(wc -l < "$MANIFEST") - 1 ))
if [ "$N" -le 0 ]; then echo 'empty manifest' >&2; exit 1; fi
export ALL_EMB_MANIFEST="$MANIFEST"
echo "Submitting $N matrix/year/mode folds"
MAX_ARRAY=1000
if [ "$N" -le "$MAX_ARRAY" ]; then
  sbatch --array="0-$((N-1))%64" --export=ALL_EMB_MANIFEST="$MANIFEST" scripts/slurm_all_embedding_cluster_regression.sbatch
else
  sbatch --array="0-$((MAX_ARRAY-1))%64" --export=ALL_EMB_MANIFEST="$MANIFEST",ARRAY_OFFSET=0 scripts/slurm_all_embedding_cluster_regression.sbatch
  REMAINING=$((N-MAX_ARRAY))
  sbatch --array="0-$((REMAINING-1))%64" --export=ALL_EMB_MANIFEST="$MANIFEST",ARRAY_OFFSET="$MAX_ARRAY" scripts/slurm_all_embedding_cluster_regression.sbatch
fi
