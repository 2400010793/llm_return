#!/bin/bash
#SBATCH --job-name=positive-downstream
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/positive_prompt_downstream/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/positive_prompt_downstream/%A_%a.err
set -euo pipefail
: "${MANIFEST:?MANIFEST required}"
EXPECTED_ROWS=${EXPECTED_ROWS:-75894}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
IFS=$'\t' read -r kind dataset prompt model variant representation target panel matrix metadata embed_model output <<< "${line%$'\r'}"
cd /mnt/lustre3/home/gaozh/llm_return
mkdir -p "$output"
if [[ "$kind" == classification ]]; then
  .venv/bin/python scripts/run_precomputed_embedding_classification.py "$panel" \
    --matrix "$matrix" --metadata "$metadata" --embedding-model "$embed_model" \
    --input-name "positive_${prompt}_${representation}" --classifier logistic \
    --reducer pca --reducer-components 128 --scaler standard \
    --train-target-column "$target" --evaluation-target-column "$target" \
    --panel-row-index-column row_index --metadata-row-index-column row_index \
    --metadata-row-index-offset 0 --expected-rows "$EXPECTED_ROWS" --search-stage fine \
    --output "$output/predictions.parquet" --artifact-dir "$output/artifacts"
else
  regression_variant="prompt_short"
  [[ "$variant" == masked_short ]] && regression_variant="masked_short"
  .venv/bin/python scripts/run_sina_precomputed_regression.py "$panel" \
    --matrix "$matrix" --metadata "$metadata" --embedding-model "$embed_model" \
    --variant "$regression_variant" --target-column "$target" --panel-row-index-column row_index \
    --metadata-row-index-column row_index --metadata-row-index-offset 0 \
    --reducer pca --reducer-components 128 --regressor ridge --run-mode final-test \
    --expected-rows "$EXPECTED_ROWS" --output "$output/stock_day_predictions.parquet" \
    --artifact-dir "$output/artifacts"
fi
