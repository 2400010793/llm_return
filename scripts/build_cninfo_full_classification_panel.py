"""Build a row-aligned classification panel for all pooled CNINFO embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_cninfo_100_classification_panel import attach_returns, code_with_exchange


def load_minimal_records(path: Path) -> pd.DataFrame:
    """Read only fields needed for labels/auditing and preserve 1-based row IDs."""
    rows: list[dict[str, object]] = []
    row_index = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row_index += 1
            record = json.loads(line)
            text = str(record.get("text_model", record.get("text", "")) or "")
            rows.append({
                "row_index": row_index,
                "document_id": record.get("document_id"),
                "stock_id": str(record.get("stock_id", "") or "").zfill(6),
                "stock_name": record.get("stock_name"),
                "announcement_date": record.get("announcement_date"),
                "published_at": record.get("published_at"),
                "title": record.get("title_clean_final", record.get("title")),
                "text_chars": len(text),
                "low_quality_text": bool(record.get("low_quality_text", False)),
                "url": record.get("url"),
                "content_type": record.get("content_type"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/cleaned/cninfo_announcements_final.jsonl"),
    )
    parser.add_argument("--returns", type=Path, default=Path("/home/team/ret.parquet"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cninfo_full_classification_panel.parquet"),
    )
    args = parser.parse_args()

    frame = load_minimal_records(args.input)
    if frame.empty:
        raise ValueError("input contains no records")
    if frame["document_id"].isna().any() or frame["document_id"].astype(str).str.strip().eq("").any():
        raise ValueError("document_id must be present and non-empty for every announcement")
    if frame["document_id"].duplicated().any():
        examples = frame.loc[frame["document_id"].duplicated(keep=False), "document_id"].head(5).tolist()
        raise ValueError(f"document_id must be unique; duplicate examples: {examples}")
    if frame["row_index"].duplicated().any():
        raise ValueError("row_index must be unique")
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True).dt.tz_convert(None)
    fallback = pd.to_datetime(frame["announcement_date"], errors="coerce")
    frame["published_at"] = frame["published_at"].fillna(fallback)
    frame["return_ticker"] = frame["stock_id"].map(code_with_exchange)
    result = attach_returns(frame, pd.read_parquet(args.returns))
    result["eligible_text_50"] = result["text_chars"].ge(50)
    result["eligible_text_100"] = result["text_chars"].ge(100)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    summary = {
        "input": str(args.input),
        "returns": str(args.returns),
        "output": str(args.output),
        "rows": int(len(result)),
        "row_index_min": int(result["row_index"].min()),
        "row_index_max": int(result["row_index"].max()),
        "document_id_unique": int(result["document_id"].nunique()),
        "stocks": int(result["stock_id"].nunique()),
        "entry_date_valid": int(result["entry_date"].notna().sum()),
        "event_label_valid": int(result["event_label"].notna().sum()),
        "next_day_label_valid": int(result["next_day_label"].notna().sum()),
        "eligible_text_50": int(result["eligible_text_50"].sum()),
        "eligible_text_100": int(result["eligible_text_100"].sum()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()