#!/bin/bash
#SBATCH --job-name=zh-embed
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-1%2
#SBATCH --output=/home/team/llm_return/logs/slurm_zh_embedding/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_zh_embedding/%A_%a.err

set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_zh_embedding data/processed/embeddings
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/home/team/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

models=(roberta bge_m3)
names=(chinese_roberta bge_m3)
model=${models[$SLURM_ARRAY_TASK_ID]}
name=${names[$SLURM_ARRAY_TASK_ID]}
if [[ "$model" == "roberta" ]]; then
  model_name=/home/team/llm_return/models/chinese-roberta-wwm-ext
  model_arg=--roberta-model
  model_flag=--roberta
else
  model_name=/home/team/llm_return/models/bge-m3
  model_arg=--bge-m3-model
  model_flag=--bge-m3
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

/usr/bin/python3 scripts/run_text_representations.py \
  data/processed/cninfo_100_classification_panel.parquet \
  --output-dir "data/processed/embeddings/${name}" \
  --text-column text \
  "${model_flag}" \
  "${model_arg}" "${model_name}" \
  --transformer-batch-size 16 \
  --max-length 256 \
  --max-input-chars 12000
