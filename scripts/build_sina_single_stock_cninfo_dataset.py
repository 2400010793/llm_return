"""Extract an aligned single-stock Sina clean corpus and label panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_single_stock_dataset(
    clean: pd.DataFrame, panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"article_id", "stock_id", "is_multi_stock_article"}
    for name, frame in (("clean", clean), ("panel", panel)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} input missing columns: {sorted(missing)}")
        if frame["article_id"].isna().any():
            raise ValueError(f"{name} article_id contains null values")

    clean_single = clean.loc[~clean["is_multi_stock_article"].fillna(True)].copy()
    panel_single = panel.loc[~panel["is_multi_stock_article"].fillna(True)].copy()
    for name, frame in (("clean", clean_single), ("panel", panel_single)):
        if frame["article_id"].duplicated().any():
            raise ValueError(f"{name} single-stock article_id must be unique")

    clean_keys = set(clean_single["article_id"].astype(str))
    panel_keys = set(panel_single["article_id"].astype(str))
    if clean_keys != panel_keys:
        raise ValueError(
            "single-stock clean/panel article sets differ: "
            f"clean_only={len(clean_keys - panel_keys)}, "
            f"panel_only={len(panel_keys - clean_keys)}"
        )

    order_column = "row_index" if "row_index" in panel_single else "article_id"
    panel_single = panel_single.sort_values(order_column, kind="stable").reset_index(drop=True)
    if "row_index" in panel_single:
        panel_single["source_row_index"] = panel_single["row_index"].astype("int64")
        panel_single["row_index"] = range(1, len(panel_single) + 1)
    else:
        panel_single.insert(0, "row_index", range(1, len(panel_single) + 1))

    clean_single = (
        clean_single.set_index("article_id", drop=False)
        .loc[panel_single["article_id"].astype(str)]
        .reset_index(drop=True)
    )
    clean_pairs = clean_single[["article_id", "stock_id"]].astype(str).reset_index(drop=True)
    panel_pairs = panel_single[["article_id", "stock_id"]].astype(str).reset_index(drop=True)
    if not clean_pairs.equals(panel_pairs):
        raise ValueError("single-stock clean/panel article and stock alignment differs")
    if len(clean_single) != len(panel_single):
        raise AssertionError("aligned clean and panel row counts differ")
    return clean_single, panel_single


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-clean", type=Path, required=True)
    parser.add_argument("--output-panel", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    output_clean, output_panel = build_single_stock_dataset(
        pd.read_parquet(args.clean), pd.read_parquet(args.panel)
    )
    for path in (args.output_clean, args.output_panel, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_clean.to_parquet(args.output_clean, index=False)
    output_panel.to_parquet(args.output_panel, index=False)

    valid_next = output_panel["next_day_return"].notna()
    valid_event = output_panel["event_return_3d"].notna()
    summary = {
        "format_version": "sina_single_stock_cninfo_v1",
        "source_clean": str(args.clean.resolve()),
        "source_panel": str(args.panel.resolve()),
        "output_clean": str(args.output_clean.resolve()),
        "output_panel": str(args.output_panel.resolve()),
        "rows": int(len(output_panel)),
        "unique_articles": int(output_panel["article_id"].nunique()),
        "stocks": int(output_panel["stock_id"].nunique()),
        "next_day_valid": int(valid_next.sum()),
        "event_3d_valid": int(valid_event.sum()),
        "all_rows_single_stock": bool((~output_panel["is_multi_stock_article"]).all()),
        "row_index_contiguous": bool(
            output_panel["row_index"].tolist() == list(range(1, len(output_panel) + 1))
        ),
        "primary_protocol": "next_day_return train -> next_day_return evaluation",
        "ablation_protocol": "event_return_3d train -> next_day_return evaluation",
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
