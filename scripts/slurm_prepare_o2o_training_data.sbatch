#!/bin/bash
#SBATCH --job-name=prepare-o2o-data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_o2o/%j.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_o2o/%j.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?O2O preparation must be submitted through task_tracker.py}"

SOURCE_PANEL=${SOURCE_PANEL:-data/processed/cninfo_full_classification_panel.parquet}
MARKET_OUTPUT=${MARKET_OUTPUT:-data/processed/cninfo_full_o2o_market.parquet}
O2O_PANEL=${O2O_PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
O2O_CACHE=${O2O_CACHE:-data/interim/o2o_training_ohlc_qfq}
O2O_START_DATE=${O2O_START_DATE:-2017-12-01}
O2O_END_DATE=${O2O_END_DATE:-2026-08-08}

mkdir -p logs/slurm_o2o "${O2O_CACHE}" "$(dirname "${MARKET_OUTPUT}")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 scripts/collect_strategy_ohlc.py \
  "${SOURCE_PANEL}" \
  --output "${MARKET_OUTPUT}" \
  --cache-dir "${O2O_CACHE}" \
  --start-date "${O2O_START_DATE}" \
  --end-date "${O2O_END_DATE}" \
  --minimum-listing-days 5 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --retries 5 \
  --request-timeout 30 \
  --provider tencent

/usr/bin/python3 scripts/build_o2o_training_panel.py \
  "${SOURCE_PANEL}" \
  --market-data "${MARKET_OUTPUT}" \
  --output "${O2O_PANEL}"

/usr/bin/python3 - "${SOURCE_PANEL}" "${O2O_PANEL}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

source_path, output_path = map(Path, sys.argv[1:])
source = pd.read_parquet(source_path, columns=["row_index"])
panel = pd.read_parquet(
    output_path,
    columns=["row_index", "entry_date", "next_day_open_to_open_return"],
)
if len(panel) != len(source):
    raise SystemExit(f"O2O panel row mismatch: {len(panel)} != {len(source)}")
if not panel["row_index"].equals(source["row_index"]):
    raise SystemExit("O2O panel changed row_index order")
years = pd.to_datetime(panel["entry_date"], errors="coerce").dt.year
coverage = panel["next_day_open_to_open_return"].notna().groupby(years).mean()
expected_years = set(range(2018, 2027))
if set(int(year) for year in coverage.index).difference(expected_years):
    raise SystemExit(f"unexpected O2O years: {coverage.to_dict()}")
missing_years = expected_years.difference(int(year) for year in coverage.index)
if missing_years:
    raise SystemExit(f"missing O2O years: {sorted(missing_years)}")
if float(coverage.loc[2018:2025].min()) < 0.80:
    raise SystemExit(f"historical O2O coverage below 80%: {coverage.to_dict()}")
print(json.dumps({
    "rows": len(panel),
    "row_index_preserved": True,
    "coverage_by_year": {str(int(k)): float(v) for k, v in coverage.items()},
}, ensure_ascii=False))
PY
