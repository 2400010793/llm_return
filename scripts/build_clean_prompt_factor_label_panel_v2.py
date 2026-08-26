"""Clean the embedding-aligned label panel and add stable risk targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = (
    "forward_compounded_return_1d",
    "forward_compounded_return_3d",
    "forward_compounded_return_5d",
    "market_adjusted_forward_return_20d",
    "post_realized_volatility_5d",
    "volatility_jump",
    "volume_shock",
    "range_shock",
    "abnormal_event_return_3d",
    "event_abs_return_3d",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    raw = pd.read_parquet(args.input)
    panel = raw.copy()
    for column in TARGET_COLUMNS:
        if column in panel:
            values = pd.to_numeric(panel[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            panel[column] = values

    eps = 1e-4
    post = pd.to_numeric(panel["post_realized_volatility_5d"], errors="coerce")
    pre = pd.to_numeric(panel["pre_realized_volatility_20d"], errors="coerce")
    ratio = (post + eps) / (pre + eps)
    panel["volatility_jump_ratio_v2"] = ratio.clip(lower=0.01, upper=100).where(np.isfinite(ratio))
    panel["volatility_jump_log_v2"] = np.log1p(post).sub(np.log1p(pre)).where(np.isfinite(post) & np.isfinite(pre))
    panel["volatility_jump_valid_v2"] = panel["volatility_jump_ratio_v2"].notna()
    panel["pre_window_valid_20d_v2"] = pre.notna()
    panel["volume_shock_valid_v2"] = panel["volume_shock"].notna()
    panel["range_shock_valid_v2"] = panel["range_shock"].notna()
    panel["market_adjusted_return_20d_valid_v2"] = panel["market_adjusted_forward_return_20d"].notna()

    summary = {
        "format_version": "prompt_factor_label_panel_v2",
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(len(panel)),
        "row_index_unique": bool(panel["row_index"].is_unique),
        "finite_counts": {
            column: int(pd.to_numeric(panel[column], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum())
            for column in (*TARGET_COLUMNS, "volatility_jump_ratio_v2", "volatility_jump_log_v2")
            if column in panel
        },
        "invalid_counts_before_cleaning": {
            column: int((~np.isfinite(pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float))).sum())
            for column in TARGET_COLUMNS
            if column in raw.columns
        },
        "definitions": {
            "volatility_jump_ratio_v2": "clip((post_realized_volatility_5d + 1e-4) / (pre_realized_volatility_20d + 1e-4), 0.01, 100)",
            "volatility_jump_log_v2": "log1p(post_realized_volatility_5d) - log1p(pre_realized_volatility_20d)",
            "warmup_policy": "retain rows but expose pre_window_valid_20d_v2; regression filters non-finite targets per task",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(panel), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
