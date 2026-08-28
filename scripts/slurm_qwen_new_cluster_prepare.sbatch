#!/bin/bash
# Wait for new Qwen GGUF shards, then submit fixed HDBSCAN folds.
#SBATCH --job-name=qwen-new-cluster-prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/qwen_new_cluster/%j.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/qwen_new_cluster/%j.err
set -euo pipefail
cd /data/alpha_team2/shares/llm_return
ROOT=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1
EMBED=${ROOT}/qwen3_gguf_prompt_mean
PANEL=${ROOT}/classification/sina_single_stock_classification_panel.parquet
OUT=${ROOT}/qwen3_new_hdbscan_v1
mkdir -p "${OUT}" logs/qwen_new_cluster
expected=64
for variant in short masked_short; do
  while true; do
    n=$(find "${EMBED}/sina/return/${variant}" -mindepth 2 -maxdepth 2 -name COMPLETED 2>/dev/null | wc -l)
    echo "variant=${variant} completed=${n}/${expected}"
    [[ "${n}" -ge "${expected}" ]] && break
    sleep 300
  done
  .venv/bin/python scripts/build_qwen_position_matrices.py --embedding-root "${EMBED}" --panel "${PANEL}" --prompt return --variant "${variant}" --output-root "${OUT}/matrices/${variant}" --groups all
done
MANIFEST=${OUT}/fold_manifest.tsv
printf 'source\tdataset\tmodel\tprompt\tvariant\tfeature\ttest_year\tmode\tpanel\tmatrix\tmetadata\n' > "${MANIFEST}"
for variant in short masked_short; do
  for year in 2018 2019 2020 2021 2022 2023 2024 2025 2026; do
    printf 'matrix\tsina\tqwen3_embedding_8b\treturn\t%s\tprompt_mean\t%s\tfixed\t%s\t%s\t%s\n' "${variant}" "${year}" "${PANEL}" "${OUT}/matrices/${variant}/all_positions/matrix.npy" "${OUT}/matrices/${variant}/all_positions/metadata.parquet" >> "${MANIFEST}"
  done
done
job=$(sbatch --parsable --array="0-17%18" --export="ALL,CORRECT_EMB_MANIFEST=${MANIFEST},QWEN_CLUSTER_OUTPUT=${OUT}/results" scripts/slurm_qwen_new_cluster_fold.sbatch)
echo "submitted qwen fixed HDBSCAN folds=${job} manifest=${MANIFEST}"
