#!/bin/bash
# Four backbones x four Sina shards. Each model load processes all five variants.
#SBATCH --job-name=sina-cninfo-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-15%4
#SBATCH --output=/home/team/llm_return/logs/slurm_sina_cninfo_method/%A_%a_embed.out
#SBATCH --error=/home/team/llm_return/logs/slurm_sina_cninfo_method/%A_%a_embed.err

set -euo pipefail
cd /home/team/llm_return
: "${SLURM_ARRAY_TASK_ID:?run as array 0..15}"

models=(roberta bge_m3 ckip_bert xlm_roberta_large)
model=${models[$((SLURM_ARRAY_TASK_ID / 4))]}
shard=$((SLURM_ARRAY_TASK_ID % 4))
input=/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/inputs/shard-${shard}
root=/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/embeddings
destination=${root}/shard-${shard}/${model}
stage=/tmp/sina-cninfo-method-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}

mkdir -p logs/slurm_sina_cninfo_method "$(dirname "${destination}")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

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
echo "promoted=${destination}"
