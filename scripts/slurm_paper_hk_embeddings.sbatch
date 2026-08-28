#!/bin/bash
#SBATCH --job-name=paper-hk-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=/home/team/llm_return/logs/slurm_paper_hk/%A_%a_embed.out
#SBATCH --error=/home/team/llm_return/logs/slurm_paper_hk/%A_%a_embed.err

set -euo pipefail
cd /home/team/llm_return
: "${MODEL:?export MODEL=ckip_bert or xlm_roberta_large}"
: "${SLURM_ARRAY_TASK_ID:?run as an array over shards 0..31}"
case "${MODEL}" in
  ckip_bert) batch_size=8 ;;
  xlm_roberta_large) batch_size=2 ;;
  *) echo "unsupported paper HK model: ${MODEL}" >&2; exit 2 ;;
esac

shard=${SLURM_ARRAY_TASK_ID}
input="data/processed/cleaned/prompt_v3_fixed_parts/shard-${shard}"
output="data/processed/pooled_paper_hk_embeddings_v1/shard-${shard}/${MODEL}/plain"
mkdir -p logs/slurm_paper_hk "${output}"
if [[ -s "${output}/summary.json" && -s "${output}/metadata.jsonl" && -s "${output}/short_pooling.npz" ]]; then
  echo "complete output already exists: ${output}"
  exit 0
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PAPER_HK_MODEL_ROOT=${PAPER_HK_MODEL_ROOT:-/home/team/.cache/huggingface/paper_hk_models}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 scripts/run_short_pooled_embeddings.py \
  "${input}" --output-dir "${output}" \
  --model "${MODEL}" --variant plain \
  --prompt-condition no_prompt_natural \
  --batch-size "${batch_size}" --max-length 512
