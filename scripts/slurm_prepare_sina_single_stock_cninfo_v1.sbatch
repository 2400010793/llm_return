#!/bin/bash
#SBATCH --job-name=sina1-prepare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/sina_single_stock_cninfo_v1/%j_prepare.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/sina_single_stock_cninfo_v1/%j_prepare.err

set -euo pipefail
cd /data/alpha_team2/shares/llm_return

root=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1
clean_full=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/cleaned_full_2010_2026/sina_all_news_clean_expanded.parquet
panel_full=/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/classification/sina_full_classification_panel.parquet
clean=${root}/clean/sina_single_stock_clean.parquet
panel=${root}/classification/sina_single_stock_classification_panel.parquet

mkdir -p logs/sina_single_stock_cninfo_v1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

if .venv/bin/python - "${clean}" "${panel}" "${root}/dataset.summary.json" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

clean, panel, summary = map(Path, sys.argv[1:])
if not all(path.is_file() and path.stat().st_size > 0 for path in (clean, panel, summary)):
    raise SystemExit(1)
data = json.loads(summary.read_text(encoding="utf-8"))
expected = 75894
valid = (
    data.get("rows") == expected
    and data.get("unique_articles") == expected
    and data.get("all_rows_single_stock") is True
    and data.get("row_index_contiguous") is True
    and pq.ParquetFile(clean).metadata.num_rows == expected
    and pq.ParquetFile(panel).metadata.num_rows == expected
)
raise SystemExit(0 if valid else 1)
PY
then
  echo "reusing validated single-stock dataset: rows=75894"
else
  .venv/bin/python scripts/build_sina_single_stock_cninfo_dataset.py \
    --clean "${clean_full}" --panel "${panel_full}" \
    --output-clean "${clean}" --output-panel "${panel}" \
    --summary "${root}/dataset.summary.json"
fi

.venv/bin/python scripts/build_sina_cninfo_method_inputs.py \
  --clean "${clean}" --panel "${panel}" \
  --output-dir "${root}/inputs" --shards 64
