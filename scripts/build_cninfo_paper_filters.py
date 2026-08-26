"""Build non-destructive CNINFO length and rolling semantic-novelty panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pooled_embeddings import align_embeddings_to_panel, load_pooled_embeddings
from src.data.rolling_semantic_dedup import rolling_prior_cosine


def threshold_tag(value: float) -> str:
    return f"semantic_{int(round(value * 100)):03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--market-data", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="roberta")
    parser.add_argument("--variant", default="masked_short")
    parser.add_argument("--feature", default="body_mean")
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=100_000)
    parser.add_argument("--lookback-trading-days", type=int, default=5)
    parser.add_argument("--thresholds", default="0.8,0.9,0.95,0.98")
    args = parser.parse_args()

    thresholds = sorted({float(value) for value in args.thresholds.split(",") if value.strip()})
    if not thresholds or any(not 0 < value <= 1 for value in thresholds):
        raise ValueError("thresholds must lie in (0, 1]")
    panel = pd.read_parquet(args.panel)
    embeddings = load_pooled_embeddings(
        args.embedding_root, model=args.model, variant=args.variant,
        feature=args.feature, require_complete_rows=350_577, max_matrix_gib=8.0,
    )
    frame, matrix = align_embeddings_to_panel(panel, embeddings)
    if len(frame) != len(panel):
        raise ValueError(f"only {len(frame)}/{len(panel)} panel rows have embeddings")
    chars = pd.to_numeric(frame["text_chars"], errors="coerce")
    length_keep = chars.between(args.min_chars, args.max_chars).to_numpy()
    market_dates = pd.read_parquet(args.market_data, columns=["entry_date"])["entry_date"]
    maximum, source_row = rolling_prior_cosine(
        frame, matrix, trading_dates=market_dates, eligible=length_keep,
        lookback_trading_days=args.lookback_trading_days,
    )
    frame["paper_length_keep"] = length_keep
    frame["rolling_prior_cosine"] = maximum
    frame["rolling_duplicate_source_row"] = source_row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    novelty_path = args.output_dir / "cninfo_rolling_semantic_novelty.parquet"
    frame[[
        "row_index", "stock_id", "entry_date", "paper_length_keep",
        "rolling_prior_cosine", "rolling_duplicate_source_row",
    ]].to_parquet(novelty_path, index=False)

    panels: dict[str, dict[str, object]] = {}
    length_path = args.output_dir / "cninfo_full_o2o_paper_length.parquet"
    frame.loc[length_keep].to_parquet(length_path, index=False)
    panels["length"] = {"path": str(length_path), "rows": int(length_keep.sum())}
    for threshold in thresholds:
        semantic_keep = length_keep & (~np.isfinite(maximum) | (maximum < threshold))
        tag = threshold_tag(threshold)
        path = args.output_dir / f"cninfo_full_o2o_{tag}.parquet"
        frame.loc[semantic_keep].to_parquet(path, index=False)
        panels[tag] = {
            "path": str(path), "threshold": threshold,
            "rows": int(semantic_keep.sum()),
            "semantic_duplicates_removed_after_length": int(length_keep.sum() - semantic_keep.sum()),
        }

    finite = maximum[np.isfinite(maximum)]
    years = pd.to_datetime(frame["entry_date"], errors="coerce").dt.year
    summary = {
        "format_version": "cninfo_paper_filter_ablation_v1",
        "input_panel": str(args.panel),
        "market_data": str(args.market_data),
        "embedding": {
            "root": str(args.embedding_root), "model": args.model,
            "variant": args.variant, "feature": args.feature,
            "note": "rolling transformer-embedding cosine; not identical to the paper's bag-of-words cosine",
        },
        "rows": int(len(frame)),
        "length_filter": {
            "min_chars": args.min_chars, "max_chars": args.max_chars,
            "kept": int(length_keep.sum()), "removed": int((~length_keep).sum()),
        },
        "lookback_trading_days": args.lookback_trading_days,
        "similarity_quantiles": (
            {str(q): float(np.quantile(finite, q)) for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)}
            if len(finite) else {}
        ),
        "panels": panels,
        "yearly_length_kept": {
            str(int(year)): int((length_keep & years.eq(year).to_numpy()).sum())
            for year in sorted(years.dropna().unique())
        },
        "novelty_output": str(novelty_path),
    }
    summary_path = args.output_dir / "cninfo_paper_filters.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
