#!/bin/bash
# Four-way CPU probe: two models x raw/masked short prompt, 256 rows each.
# The probe validates shapes, fixed prompt slots, offsets, and output size
# before the 32-way full run is submitted.
#SBATCH --job-name=short-token-probe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_short_token_probe/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_short_token_probe/%A_%a.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_short_token_probe data/processed/token_embeddings_probe_v3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
case "$SLURM_ARRAY_TASK_ID" in
  0) model=roberta; variant=short; max_length=512; name=roberta_short ;;
  1) model=roberta; variant=masked_short; max_length=512; name=roberta_masked_short ;;
  2) model=bge_m3; variant=short; max_length=1000; name=bge_m3_short ;;
  3) model=bge_m3; variant=masked_short; max_length=1000; name=bge_m3_masked_short ;;
esac
/usr/bin/python3 scripts/run_short_token_embeddings.py \
  data/processed/cleaned/prompt_v3_fixed_parts \
  --output-dir "data/processed/token_embeddings_probe_v3/${name}" \
  --model "$model" --variant "$variant" --rows 256 \
  --batch-size 4 --max-length "$max_length"