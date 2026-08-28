#!/usr/bin/env python3
"""Build Token/Body cluster jobs for newly added target labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE = Path("/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/aligned_masked_short_v1")
TARGETS = {
    "valuation": ["next_pe_log_dev", "next_pb_log_dev", "next_ps_log_dev", "next_evtoebitda_log_dev"],
    "volatility": ["volatility_jump_log_v2", "next_intraday_rvol_log_change", "next_intraday_rvol_close"],
    "liquidity": ["next_intraday_spread_log_change", "next_intraday_spread_bps_mean"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    panel_columns = set(pd.read_parquet(args.panel, columns=None).columns)
    rows = []
    for axis, targets in TARGETS.items():
        for target in targets:
            if target not in panel_columns:
                raise ValueError(f"target missing from panel: {target}")
            for representation, factor_root, factor_name in (
                ("token", BASE / "factors_token", "target_span_mean"),
                ("body", BASE / "factors", "body_mean"),
            ):
                matrix = factor_root / "roberta" / axis / "masked_short" / factor_name / "direction.npy"
                metadata = factor_root / "roberta" / axis / "masked_short" / factor_name / "metadata.parquet"
                if not matrix.is_file() or not metadata.is_file():
                    raise FileNotFoundError(f"missing factor matrix: {matrix}")
                rows.append({
                    "axis": axis,
                    "representation": representation,
                    "target": target,
                    "panel": str(args.panel),
                    "matrix": str(matrix),
                    "metadata": str(metadata),
                    "output": str(args.output_root / f"{axis}_{representation}_{target}.csv"),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"wrote {len(rows)} cluster tasks to {args.output}")


if __name__ == "__main__":
    main()
