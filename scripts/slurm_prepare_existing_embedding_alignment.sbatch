#!/bin/bash
#SBATCH --job-name=prep-embed-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%12
#SBATCH --output=/home/team/llm_return/logs/slurm_prep_embed_cls/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_prep_embed_cls/%A_%a.err
set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_prep_embed_cls data/processed/classification_embeddings reports/data_quality
models=(chinese_roberta chinese_roberta chinese_roberta chinese_roberta chinese_roberta bge_m3 bge_m3 bge_m3 bge_m3 bge_m3 word2vec word2vec)
inputs=(text_plain text_input_2_short text_input_3_long text_input_5_masked_short text_input_6_masked_long text_plain text_input_2_short text_input_3_long text_input_5_masked_short text_input_6_masked_long text_input_5_masked_short text_input_6_masked_long)
model=${models[$SLURM_ARRAY_TASK_ID]}; input=${inputs[$SLURM_ARRAY_TASK_ID]}
emb="data/processed/embeddings/all_inputs/${model}/${input}"
name="${model}_${input}"
/usr/bin/python3 scripts/prepare_existing_embedding_classification.py \
 --panel data/processed/cninfo_100_classification_panel.parquet \
 --cleaned data/processed/cleaned/cninfo_announcements_final.jsonl \
 --embedding "${emb}/${model}.npy" --metadata "${emb}/metadata.parquet" \
 --out-panel "data/processed/classification_embeddings/${name}.parquet" \
 --out-npy "data/processed/classification_embeddings/${name}.npy" \
 --report "reports/data_quality/${name}_alignment.json"
