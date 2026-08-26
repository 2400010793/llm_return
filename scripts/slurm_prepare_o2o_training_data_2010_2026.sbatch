#!/bin/bash
# Rebuild strict O2O labels for the complete 2010-2026 text panel.
# Uses Tencent HFQ because historical Tencent QFQ responses can contain
# non-positive adjusted prices. Existing 2018-2026 outputs are untouched.
#SBATCH --job-name=o2o-2010-2026
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_o2o_2010_2026/%j.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_o2o_2010_2026/%j.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?O2O preparation must be submitted through task_tracker.py}"
SOURCE_PANEL=${SOURCE_PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
MARKET_OUTPUT=${MARKET_OUTPUT:-data/processed/cninfo_full_o2o_market_2010_2026_hfq.parquet}
O2O_PANEL=${O2O_PANEL:-data/processed/cninfo_full_o2o_panel_2010_2026_hfq.parquet}
O2O_CACHE=${O2O_CACHE:-data/interim/o2o_training_ohlc_hfq_2010_2026}
O2O_START_DATE=${O2O_START_DATE:-2009-12-01}
O2O_END_DATE=${O2O_END_DATE:-2026-08-08}

mkdir -p logs/slurm_o2o_2010_2026 "${O2O_CACHE}" "$(dirname "${MARKET_OUTPUT}")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 scripts/collect_strategy_ohlc.py "${SOURCE_PANEL}" \
  --output "${MARKET_OUTPUT}" --cache-dir "${O2O_CACHE}" \
  --start-date "${O2O_START_DATE}" --end-date "${O2O_END_DATE}" \
  --minimum-listing-days 5 --workers "${SLURM_CPUS_PER_TASK}" \
  --retries 5 --request-timeout 30 --provider tencent --adjust hfq

MARKET_SUMMARY="${MARKET_OUTPUT%.*}.summary.json"
/usr/bin/python3 - "${MARKET_SUMMARY}" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary["failures"]:
    raise SystemExit(f"market collection failures: {summary['failures']}")
if summary["collected_stocks"] != summary["requested_stocks"]:
    raise SystemExit("market collection did not cover the full stock universe")
if summary.get("adjust") != "hfq":
    raise SystemExit(f"unexpected adjustment mode: {summary.get('adjust')}")
print(json.dumps({"stocks": summary["collected_stocks"], "adjust": summary["adjust"]}))
PY

/usr/bin/python3 scripts/build_o2o_training_panel.py "${SOURCE_PANEL}" \
  --market-data "${MARKET_OUTPUT}" --output "${O2O_PANEL}"

/usr/bin/python3 - "${SOURCE_PANEL}" "${MARKET_OUTPUT}" "${O2O_PANEL}" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
source_path, market_path, output_path = map(Path, sys.argv[1:])
source = pd.read_parquet(source_path, columns=["row_index", "entry_date"])
market = pd.read_parquet(market_path, columns=["open", "close", "high", "low", "stock_id"])
panel = pd.read_parquet(output_path, columns=["row_index", "entry_date", "next_day_open_to_open_return"])
prices = market[["open", "close", "high", "low"]].apply(pd.to_numeric, errors="coerce")
if (prices <= 0).any().any():
    raise SystemExit("non-positive prices remain in the cleaned market panel")
if len(panel) != len(source) or not panel["row_index"].equals(source["row_index"]):
    raise SystemExit("O2O panel row_index alignment failed")
years = pd.to_datetime(panel["entry_date"], errors="coerce").dt.year
coverage = panel["next_day_open_to_open_return"].notna().groupby(years).mean()
expected = set(range(2010, 2027))
if set(int(y) for y in coverage.index) != expected:
    raise SystemExit(f"unexpected O2O years: {coverage.to_dict()}")
print(json.dumps({"rows": len(panel), "stocks": int(market["stock_id"].nunique()),
                  "market_rows": len(market),
                  "coverage_by_year": {str(int(k)): float(v) for k, v in coverage.items()},
                  "min_price": float(prices.min().min())}, ensure_ascii=False))
PY
