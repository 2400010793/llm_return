#!/bin/bash
#SBATCH --job-name=aligned-factor-matched
#SBATCH --partition=cpu
#SBATCH --array=0-11%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/aligned_factor_matched/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/aligned_factor_matched/%A_%a.err
set -euo pipefail
REPO=/mnt/lustre3/home/gaozh/llm_return
BASE=/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/aligned_masked_short_v1
PANEL=${REPO}/data/processed/prompt_factor_labels/prompt_factor_label_panel_v2.parquet
AXES=(horizon valuation certainty volatility impact liquidity)
TARGETS=(forward_compounded_return_5d market_adjusted_forward_return_20d post_realized_volatility_5d volatility_jump_log_v2 event_abs_return_3d volume_shock)
REPS=(token body)
axis_index=$((SLURM_ARRAY_TASK_ID / 2))
rep_index=$((SLURM_ARRAY_TASK_ID % 2))
axis=${AXES[$axis_index]}
target=${TARGETS[$axis_index]}
rep=${REPS[$rep_index]}
if [[ ${rep} == token ]]; then
  matrix=${BASE}/factors_token/roberta/${axis}/masked_short/target_span_mean/direction.npy
  metadata=${BASE}/factors_token/roberta/${axis}/masked_short/target_span_mean/metadata.parquet
else
  matrix=${BASE}/factors/roberta/${axis}/masked_short/body_mean/direction.npy
  metadata=${BASE}/factors/roberta/${axis}/masked_short/body_mean/metadata.parquet
fi
out=${REPO}/reports/aligned_factor_clusters/${axis}_${rep}_matched.csv
mkdir -p ${REPO}/logs/aligned_factor_matched ${REPO}/reports/aligned_factor_clusters
[[ -s ${out} ]] && exit 0
cd ${REPO}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/run_aligned_factor_cluster_comparison.py \
  --panel ${PANEL} --matrix ${matrix} --metadata ${metadata} \
  --output ${out} --axis ${axis} --representation ${rep} \
  --target ${target} --first-year 2026 --last-year 2026 --sample-rows 20000
