#!/usr/bin/env python3
"""Clean a CSV/Parquet news table for auditable paper-style experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.clean_paper_news import clean_news_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--max-chars", type=int, default=50000)
    parser.add_argument("--stopwords", default="")
    args = parser.parse_args()
    source = Path(args.input)
    if source.is_dir():
        records = []
        for path in sorted(source.glob("*.json")):
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            records.extend(payload.get("records", []))
        frame = pd.DataFrame(records)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        frame = pd.read_parquet(source)
    stopwords = set(Path(args.stopwords).read_text(encoding="utf-8").split()) if args.stopwords else None
    clean, stats = clean_news_frame(frame, min_chars=args.min_chars, max_chars=args.max_chars, stopwords=stopwords)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".csv":
        clean.to_csv(destination, index=False)
    else:
        clean.to_parquet(destination, index=False)
    summary = {"input": str(source), "output": str(destination), **stats}
    destination.with_suffix(destination.suffix + ".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
