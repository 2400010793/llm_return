#!/bin/bash
# One length group per submission: 32 shards x 2 models; raw+masked sequentially.
#SBATCH --job-name=prompt-token-v5
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --array=0-63%4
#SBATCH --output=/home/team/llm_return/logs/slurm_prompt_token_v5/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_prompt_token_v5/%A_%a.err

set -euo pipefail
cd /home/team/llm_return
: "${LENGTH_GROUP:?export LENGTH_GROUP=short or long}"
mkdir -p logs/slurm_prompt_token_v5 data/processed/prompt_token_embeddings_v5
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

shard=$((SLURM_ARRAY_TASK_ID / 2))
model_id=$((SLURM_ARRAY_TASK_ID % 2))
case "${model_id}" in
  0) model=roberta; max_length=512 ;;
  1) model=bge_m3; max_length=1000 ;;
  *) echo "invalid model id" >&2; exit 2 ;;
esac
case "${LENGTH_GROUP}" in
  short)
    input="data/processed/cleaned/prompt_v3_fixed_parts/shard-${shard}"
    variants=(short masked_short)
    ;;
  long)
    input="data/processed/cleaned/prompt_v4_fixed_long_parts/shard-${shard}"
    variants=(long masked_long)
    ;;
  *) echo "invalid LENGTH_GROUP=${LENGTH_GROUP}" >&2; exit 2 ;;
esac

for variant in "${variants[@]}"; do
  output="data/processed/prompt_token_embeddings_v5/shard-${shard}/${model}/${variant}"
  /usr/bin/python3 scripts/run_prompt_token_embeddings_v5.py \
    "${input}" --output-dir "${output}" \
    --model "${model}" --variant "${variant}" \
    --batch-size 4 --max-length "${max_length}"
done