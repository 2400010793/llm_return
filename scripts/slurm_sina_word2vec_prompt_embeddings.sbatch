#!/bin/bash
# Encode the five Sina prompt/mask variants with the existing Word2Vec backend.
# Array mapping: one Word2Vec model trained per text variant. Qwen is excluded.
#SBATCH --job-name=sina-word2vec-prompt
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-4%2
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_sina_word2vec_prompt/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_sina_word2vec_prompt/%A_%a.err

set -euo pipefail

REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
DATA_ROOT=/home/gaozh/news_content_quality_20260812
INPUT=${DATA_ROOT}/prompts/sina_single_stock_prompt_variants.parquet
RELEASE_ROOT=${DATA_ROOT}/embeddings/single_stock/word2vec_prompt
STAGE=/tmp/sina-word2vec-prompt-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}

cd "${REPO_ROOT}"
mkdir -p logs/slurm_sina_word2vec_prompt "${RELEASE_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
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
echo "model=word2vec variant=${variant} output=${DEST}"

.venv/bin/python scripts/run_text_representations.py "${INPUT}" \
  --output-dir "${OUTPUT}" \
  --text-column "${variant}" \
  --word2vec \
  --max-input-chars 12000

.venv/bin/python scripts/audit_embedding_matrix.py \
  --input "${INPUT}" \
  --text-column "${variant}" \
  --matrix "${OUTPUT}/word2vec.npy" \
  --metadata "${OUTPUT}/metadata.parquet" \
  --output "${OUTPUT}/audit.json" \
  --expected-dimensions 100 \
  --max-input-chars 12000 \
  --allow-duplicate-vectors

cp -a "${OUTPUT}" "${INCOMING}"
mv "${INCOMING}" "${DEST}"
echo "promoted=${DEST}"