#!/bin/bash
# Four CNINFO-standard backbones x 64 shards; each task emits all five variants.
#SBATCH --job-name=sina1-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-255%16
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/sina_single_stock_cninfo_v1/%A_%a_embed.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/sina_single_stock_cninfo_v1/%A_%a_embed.err

set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
: "${SLURM_ARRAY_TASK_ID:?run as array 0..255}"

models=(roberta bge_m3 ckip_bert xlm_roberta_large)
model=${models[$((SLURM_ARRAY_TASK_ID / 64))]}
shard=$((SLURM_ARRAY_TASK_ID % 64))
root=/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1
input=${root}/inputs/shard-${shard}
destination=${root}/embeddings/shard-${shard}/${model}
stage=${SLURM_TMPDIR:-/tmp}/sina1-cninfo-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}

mkdir -p logs/sina_single_stock_cninfo_v1 "$(dirname "${destination}")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=/mnt/lustre3/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

complete=1
for variant in plain short masked_short long masked_long; do
  for file in summary.json metadata.jsonl short_pooling.npz; do
    [[ -s "${destination}/${variant}/${file}" ]] || complete=0
  done
  if [[ "${variant}" != plain ]]; then
    [[ -s "${destination}/${variant}/prompt_token_embeddings.npy" ]] || complete=0
  fi
done
if (( complete )); then
  echo "complete output already exists: ${destination}"
  exit 0
fi
if [[ -e "${destination}" ]]; then
  echo "refusing to overwrite incomplete destination: ${destination}" >&2
  exit 1
fi
cleanup() { rm -rf "${stage}"; }
trap cleanup EXIT

case "${model}" in
  bge_m3) max_length=1000; batch_size=4 ;;
  xlm_roberta_large) max_length=512; batch_size=2 ;;
  *) max_length=512; batch_size=4 ;;
esac

.venv/bin/python scripts/run_sina_cninfo_method_embeddings.py "${input}" \
  --output-dir "${stage}" --model "${model}" \
  --batch-size "${batch_size}" --max-length "${max_length}" --device cpu

incoming=$(dirname "${destination}")/.${model}.${SLURM_JOB_ID}.release
mv "${stage}" "${incoming}"
mv "${incoming}" "${destination}"
echo "promoted=${destination} model=${model} shard=${shard} device=cpu"
