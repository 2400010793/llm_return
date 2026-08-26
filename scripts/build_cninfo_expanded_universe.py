"""Build a current CNINFO universe whose stocks have downstream return labels."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import akshare as ak
import pandas as pd


def exchange_for(code: str) -> str:
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "SH"
    if code.startswith(("000", "001", "002", "003", "300", "301", "302")):
        return "SZ"
    return "BJ"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--returns", type=Path, default=Path("/mnt/lustre3/home/gaozh/ret.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/stock_universe_cninfo_current_with_returns.csv"))
    parser.add_argument("--source-snapshot", type=Path, default=Path("data/source_stock_lists/a_share_code_name_20260818.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/cninfo_expanded_universe_20260818.json"))
    args = parser.parse_args()

    current = ak.stock_info_a_code_name().rename(columns={"code": "stock_id", "name": "stock_name"})
    current["stock_id"] = current["stock_id"].astype(str).str.strip().str.zfill(6)
    current["stock_name"] = current["stock_name"].astype(str).str.strip()
    current = current[current["stock_id"].str.fullmatch(r"\d{6}")].drop_duplicates("stock_id")
    current["exchange"] = current["stock_id"].map(exchange_for)

    returns = pd.read_parquet(args.returns)
    return_ids = {
        match.group(1)
        for column in returns.columns
        if (match := re.fullmatch(r"(\d{6})\.(?:SH|SZ)", str(column)))
    }
    selected = current[
        current["exchange"].isin(["SH", "SZ"]) & current["stock_id"].isin(return_ids)
    ].copy()
    selected["industry"] = ""
    selected["xueqiu_symbol"] = selected["exchange"] + selected["stock_id"]
    selected["active"] = "1"
    selected = selected[
        ["stock_id", "stock_name", "exchange", "industry", "xueqiu_symbol", "active"]
    ].sort_values(["exchange", "stock_id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.source_snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(args.source_snapshot, index=False, encoding="utf-8")
    selected.to_csv(args.output, index=False, encoding="utf-8")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_a_share_stocks": len(current),
        "return_columns": len(return_ids),
        "selected_stocks": len(selected),
        "selected_by_exchange": selected["exchange"].value_counts().sort_index().to_dict(),
        "excluded_bj_without_returns": int((current["exchange"] == "BJ").sum()),
        "current_sh_sz_without_returns": int(
            (current["exchange"].isin(["SH", "SZ"]) & ~current["stock_id"].isin(return_ids)).sum()
        ),
        "source": "AkShare stock_info_a_code_name",
        "returns": str(args.returns),
        "output": str(args.output),
        "source_snapshot": str(args.source_snapshot),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
