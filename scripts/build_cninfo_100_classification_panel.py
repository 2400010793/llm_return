"""Build a clean 100-stock CNINFO announcement classification panel.

The existing paper-100 universe is used as the cross-industry sample. CNINFO
records are filtered to that universe, PDF text is cleaned, publication dates
are aligned to the next trading day, and close-to-close forward returns are
computed from the available daily return matrix. No price or text information
from after the announcement is used to construct the text observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", " ").replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"第\s*\d+\s*页\s*(?:共\s*\d+\s*页)?", " ", text, flags=re.I)
    text = re.sub(r"(?:www\.|https?://)\S+", " ", text)
    return text.strip()


def parse_time(frame: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(frame.get("published_at"), errors="coerce", utc=True)
    announcement = pd.to_datetime(frame.get("announcement_date"), errors="coerce", utc=True)
    return parsed.fillna(announcement).dt.tz_convert(None)


def load_records(source_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for path in sorted(source_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("records", payload if isinstance(payload, list) else [])
            if isinstance(values, list):
                records.extend(values)
        except (OSError, json.JSONDecodeError):
            continue
    return pd.DataFrame(records)


def code_with_exchange(code: object) -> str | None:
    value = str(code or "").strip().zfill(6)
    if not re.fullmatch(r"\d{6}", value):
        return None
    return f"{value}.{'SH' if value.startswith(('5', '6')) else 'SZ'}"


def attach_returns(frame: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    prices = returns.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce").tz_localize(None).normalize()
    prices = prices[~prices.index.isna()].sort_index()
    trading_dates = pd.DatetimeIndex(prices.index.drop_duplicates())
    frame = frame.copy()
    frame["event_center_date"] = frame["published_at"].map(
        lambda value: trading_dates[trading_dates >= value.normalize()][0]
        if pd.notna(value) and (trading_dates >= value.normalize()).any()
        else pd.NaT
    )
    frame["entry_date"] = frame["published_at"].map(
        lambda value: trading_dates[trading_dates > value.normalize()][0]
        if pd.notna(value) and (trading_dates > value.normalize()).any() else pd.NaT
    )
    horizons = (1, 3, 5, 20)
    labels = {f"return_{h}": [] for h in horizons}
    for row in frame.itertuples(index=False):
        ticker = row.return_ticker
        entry = row.entry_date
        values = prices[ticker] if ticker in prices.columns and pd.notna(entry) else None
        for horizon in horizons:
            result = np.nan
            if values is not None:
                position = trading_dates.searchsorted(entry, side="left")
                dates = trading_dates[position : position + horizon + 1]
                if len(dates) == horizon + 1:
                    window = pd.to_numeric(values.reindex(dates), errors="coerce").dropna()
                    if len(window) == horizon + 1:
                        result = float((1.0 + window.iloc[1:]).prod() - 1.0)
            labels[f"return_{horizon}"].append(result)
    for name, values in labels.items():
        frame[name] = values

    # Paper classification labels: use the three daily return observations
    # centered on the publication date for in-sample sentiment training, and
    # the first tradable day's close-to-close return for next-period testing.
    event_returns = []
    event_starts = []
    event_ends = []
    next_returns = []
    for row in frame.itertuples(index=False):
        ticker = row.return_ticker
        values = prices[ticker] if ticker in prices.columns else None
        center = row.event_center_date
        entry = row.entry_date
        event_value = np.nan
        event_start = pd.NaT
        event_end = pd.NaT
        next_value = np.nan
        if values is not None and pd.notna(center):
            center_pos = trading_dates.searchsorted(center, side="left")
            # Three daily return observations: day -1, day 0, day +1.
            event_dates = trading_dates[center_pos - 1 : center_pos + 2]
            if len(event_dates) == 3:
                window = pd.to_numeric(values.reindex(event_dates), errors="coerce")
                if window.notna().all():
                    event_value = float((1.0 + window).prod() - 1.0)
                    event_start = event_dates[0]
                    event_end = event_dates[-1]
        if values is not None and pd.notna(entry):
            next_value = pd.to_numeric(values.get(entry), errors="coerce")
            next_value = float(next_value) if pd.notna(next_value) else np.nan
        event_returns.append(event_value)
        event_starts.append(event_start)
        event_ends.append(event_end)
        next_returns.append(next_value)
    frame["event_start_date"] = event_starts
    frame["event_end_date"] = event_ends
    frame["event_return_3d"] = event_returns
    frame["event_label"] = np.where(
        pd.Series(event_returns, index=frame.index).notna(),
        (pd.Series(event_returns, index=frame.index) > 0).astype("int8"),
        np.nan,
    )
    frame["next_day_return"] = next_returns
    frame["next_day_label"] = np.where(
        pd.Series(next_returns, index=frame.index).notna(),
        (pd.Series(next_returns, index=frame.index) > 0).astype("int8"),
        np.nan,
    )
    # Keep legacy names for existing consumers, but make them use the new
    # classification-specific definitions in newly built panels.
    frame["return_3d_event"] = frame["event_return_3d"]
    frame["label_3d_direction"] = frame["event_label"]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/interim/cninfo_paper_1000_focus")
    parser.add_argument("--universe", default="data/stock_universe_paper_100.csv")
    parser.add_argument("--returns", default="/home/gaozh/ret.parquet")
    parser.add_argument("--output", default="data/processed/cninfo_100_classification_panel.parquet")
    parser.add_argument("--min-text-length", type=int, default=500)
    parser.add_argument("--max-text-length", type=int, default=50000)
    args = parser.parse_args()

    universe = pd.read_csv(args.universe, dtype={"stock_id": str})
    universe["stock_id"] = universe["stock_id"].str.zfill(6)
    allowed = set(universe.loc[universe["active"].astype(int).eq(1), "stock_id"])
    frame = load_records(Path(args.source_dir))
    if frame.empty:
        raise ValueError("no CNINFO records found")
    frame["stock_id"] = frame["stock_id"].astype(str).str.extract(r"(\d{6})", expand=False)
    frame = frame[frame["stock_id"].isin(allowed)].copy()
    frame["pdf_text"] = frame.get("pdf_text", pd.Series("", index=frame.index)).map(clean_text)
    frame["title"] = frame.get("title", pd.Series("", index=frame.index)).map(clean_text)
    frame["text"] = (frame["title"] + "\n" + frame["pdf_text"]).str.strip()
    frame["published_at"] = parse_time(frame)
    frame = frame[frame["published_at"].notna()].copy()
    frame = frame[frame["text"].str.len().between(args.min_text_length, args.max_text_length)].copy()
    frame["return_ticker"] = frame["stock_id"].map(code_with_exchange)
    frame["document_id"] = frame.apply(
        lambda row: hashlib.sha256(f"{row.get('url','')}|{row['stock_id']}".encode()).hexdigest()[:20], axis=1
    )
    frame = frame.sort_values(["stock_id", "published_at", "document_id"])
    frame = frame.drop_duplicates(subset=["stock_id", "document_id"], keep="first")
    frame = attach_returns(frame, pd.read_parquet(args.returns))
    keep = ["document_id", "stock_id", "stock_name", "announcement_date", "published_at", "event_center_date", "event_start_date", "event_end_date", "entry_date", "title", "pdf_text", "text", "url", "pdf_path", "content_type", "return_1", "return_3", "return_5", "return_20", "event_return_3d", "event_label", "next_day_return", "next_day_label", "return_3d_event", "label_3d_direction"]
    keep = [column for column in keep if column in frame.columns]
    result = frame[keep].reset_index(drop=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False)
    summary = {
        "input_records_after_universe": int(len(frame)),
        "output_rows": int(len(result)),
        "stocks": int(result["stock_id"].nunique()),
        "industries": int(universe[universe["stock_id"].isin(result["stock_id"])] ["industry"].nunique()),
        "text_min": args.min_text_length,
        "text_max": args.max_text_length,
        "entry_date_valid": int(result["entry_date"].notna().sum()),
        "event_label_valid": int(result["event_label"].notna().sum()),
        "next_day_label_valid": int(result["next_day_label"].notna().sum()),
        "event_positive_rate": float(result["event_label"].mean()),
        "next_day_positive_rate": float(result["next_day_label"].mean()),
        "output": str(destination),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
