#!/bin/bash
# Extended three-day-event training / next-day evaluation experiments.
# Array: representation x classifier x reducer.
# All reducers are fitted independently inside each rolling training window.
#SBATCH --job-name=extended-time-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-134%6
#SBATCH --output=/home/team/llm_return/logs/slurm_extended_time_cls/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_extended_time_cls/%A_%a.err

set -euo pipefail
cd /home/team/llm_return
mkdir -p reports/classification/time_aligned_extended logs/slurm_extended_time_cls
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

representations=(word_tfidf char_tfidf word_count lexicon word2vec)
classifiers=(logistic linear_svm nb_svm multinomial_nb complement_nb sgd random_forest mlp knn)
reducers=(none svd pca)
representation=${representations[$((SLURM_ARRAY_TASK_ID / 27))]}
classifier=${classifiers[$(((SLURM_ARRAY_TASK_ID / 3) % 9))]}
reducer=${reducers[$((SLURM_ARRAY_TASK_ID % 3))]}

# PCA is for dense representations; sparse text uses SVD. Skip invalid pairs.
if [[ "${reducer}" == "pca" && "${representation}" != "word2vec" && "${representation}" != "lexicon" ]]; then
  exit 0
fi
if [[ "${reducer}" == "svd" && "${representation}" != "word_tfidf" && "${representation}" != "char_tfidf" && "${representation}" != "word_count" ]]; then
  exit 0
fi
# KNN on raw high-dimensional sparse text is intentionally excluded.
if [[ "${classifier}" == "knn" && "${reducer}" == "none" && "${representation}" != "word2vec" && "${representation}" != "lexicon" ]]; then
  exit 0
fi
# Multinomial/Complement NB require non-negative, unreduced count/TF-IDF input.
if [[ "${classifier}" == "multinomial_nb" || "${classifier}" == "complement_nb" ]]; then
  if [[ "${reducer}" != "none" || "${representation}" != "word_tfidf" && "${representation}" != "char_tfidf" && "${representation}" != "word_count" ]]; then
    exit 0
  fi
fi

output="reports/classification/time_aligned_extended/${representation}_${classifier}_${reducer}_seed42.json"
/usr/bin/python3 scripts/run_paper_rolling_classification.py \
  data/processed/cninfo_100_classification_panel_time_aligned.parquet \
  --train-target-column event_return_3d \
  --evaluation-target-column next_day_return \
  --representations "${representation}" \
  --classifiers "${classifier}" \
  --reducer "${reducer}" \
  --reducer-components 128 \
  --seed 42 \
  --search-stage coarse \
  --max-input-chars 12000 \
  --max-features 50000 \
  --word2vec-size 100 \
  --word2vec-epochs 10 \
  --output "${output}"