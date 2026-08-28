#!/bin/bash
# One forward pass per backbone writes mean, CLS/BOS, and max representations.
# CPU execution avoids the currently unavailable fixed GPU nodes.
#SBATCH --job-name=sina-pooling-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-3%2
#SBATCH --output=/home/team/llm_return/logs/slurm_sina_pooling/%A_%a_embed.out
#SBATCH --error=/home/team/llm_return/logs/slurm_sina_pooling/%A_%a_embed.err

set -euo pipefail
cd /home/team/llm_return
: "${SLURM_ARRAY_TASK_ID:?run as an array over 0..3}"

models=(chinese_bert chinese_roberta bge_m3 xlm_roberta_large)
model=${models[$SLURM_ARRAY_TASK_ID]}
input=/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/prompts/sina_single_stock_prompt_variants.parquet
root=/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/embeddings/single_stock/pooling_ablation_v1
destination=${root}/${model}
stage=/tmp/sina-pooling-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}

mkdir -p logs/slurm_sina_pooling "${root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

if [[ -s "${destination}/summary.json" && -s "${destination}/mean.npy" && \
      -s "${destination}/cls.npy" && -s "${destination}/max.npy" && \
      -s "${destination}/metadata.parquet" ]]; then
  echo "complete output already exists: ${destination}"
  exit 0
fi
cleanup() { rm -rf "${stage}"; }
trap cleanup EXIT

.venv/bin/python scripts/run_sina_pooling_embeddings.py "${input}" \
  --output-dir "${stage}" --model "${model}" --max-length 512 --device cpu

incoming=${root}/.${model}.${SLURM_JOB_ID}.release
rm -rf "${incoming}"
mv "${stage}" "${incoming}"
if [[ -e "${destination}" ]]; then
  echo "refusing to overwrite incomplete destination: ${destination}" >&2
  exit 1
fi
mv "${incoming}" "${destination}"
echo "promoted=${destination}"
