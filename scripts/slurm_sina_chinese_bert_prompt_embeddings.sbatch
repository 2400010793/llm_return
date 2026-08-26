#!/bin/bash
# Encode the five Sina prompt/mask variants with the cached Chinese BERT model.
# The model is CKIPLab bert-base-chinese (BERT, hidden size 768), kept separate
# from the existing Chinese-RoBERTa outputs and from all Qwen outputs.
# Array mapping: 0 plain, 1 short prompt, 2 long prompt, 3 masked short,
# 4 masked long.
#SBATCH --job-name=sina-ckip-bert
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-4%1
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_sina_ckip_bert/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_sina_ckip_bert/%A_%a.err

set -euo pipefail

REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
DATA_ROOT=/home/gaozh/news_content_quality_20260812
INPUT=${DATA_ROOT}/prompts/sina_single_stock_prompt_variants.parquet
MODEL=/home/gaozh/.cache/huggingface/paper_hk_models/ckiplab-bert-base-chinese
RELEASE_ROOT=${DATA_ROOT}/embeddings/single_stock/chinese_bert_prompt
STAGE=/tmp/sina-ckip-bert-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}

cd "${REPO_ROOT}"
mkdir -p logs/slurm_sina_ckip_bert "${RELEASE_ROOT}"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=${HF_HOME:-/home/gaozh/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

variants=(text_plain text_prompt_short text_prompt_long text_masked_short text_masked_long)
names=(plain prompt_short prompt_long masked_short masked_long)
variant=${variants[$SLURM_ARRAY_TASK_ID]}
name=${names[$SLURM_ARRAY_TASK_ID]}
OUTPUT=${STAGE}/${name}
DEST=${RELEASE_ROOT}/${name}
INCOMING=${RELEASE_ROOT}/.${name}.${SLURM_JOB_ID}.release

if [[ -e "${DEST}" || -e "${INCOMING}" ]]; then
  echo "refusing to overwrite existing embedding directory: ${DEST}" >&2
  exit 1
fi

cleanup() {
  rm -rf "${STAGE}"
}
trap cleanup EXIT

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
echo "model=${MODEL} variant=${variant} output=${DEST}"

.venv/bin/python scripts/run_text_representations.py "${INPUT}" \
  --output-dir "${OUTPUT}" \
  --text-column "${variant}" \
  --bert --bert-model "${MODEL}" \
  --transformer-batch-size 16 \
  --max-length 512 \
  --max-input-chars 12000

.venv/bin/python scripts/audit_embedding_matrix.py \
  --input "${INPUT}" \
  --text-column "${variant}" \
  --matrix "${OUTPUT}/chinese_bert.npy" \
  --metadata "${OUTPUT}/metadata.parquet" \
  --output "${OUTPUT}/audit.json" \
  --expected-dimensions 768 \
  --max-input-chars 12000 \
  --allow-duplicate-vectors

cp -a "${OUTPUT}" "${INCOMING}"
mv "${INCOMING}" "${DEST}"
echo "promoted=${DEST}"