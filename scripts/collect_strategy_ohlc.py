"""Collect and cache adjusted A-share OHLC for a prediction universe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.market_execution import build_execution_market_panel, normalize_akshare_ohlc


def _collect_one(
    stock_id: str,
    *,
    start: str,
    end: str,
    cache_dir: Path,
    retries: int,
    request_timeout: float,
    provider: str,
    adjust: str,
) -> tuple[str, str | None]:
    destination = cache_dir / f"{stock_id}.parquet"
    if destination.exists():
        cached = pd.read_parquet(destination, columns=["date"])
        expected_last = pd.Timestamp(end) - pd.offsets.BDay(1)
        if not cached.empty and pd.to_datetime(cached["date"]).max() >= expected_last:
            return stock_id, None
    error: Exception | None = None
    for attempt in range(retries):
        try:
            if provider == "tencent":
                exchange = "sh" if stock_id.startswith(("5", "6")) else "sz"
                raw = ak.stock_zh_a_hist_tx(
                    symbol=f"{exchange}{stock_id}",
                    start_date=start.replace("-", ""),
                    end_date=end.replace("-", ""),
                    adjust=adjust,
                    timeout=request_timeout,
                )
                raw["stock_id"] = stock_id
            else:
                raw = ak.stock_zh_a_hist(
                    symbol=stock_id,
                    period="daily",
                    start_date=start.replace("-", ""),
                    end_date=end.replace("-", ""),
                    adjust=adjust,
                    timeout=request_timeout,
                )
            if raw.empty:
                raise ValueError("empty OHLC response")
            normalized = normalize_akshare_ohlc(raw)
            normalized.to_parquet(destination, index=False)
            return stock_id, None
        except Exception as exc:  # network/provider failures are retried and audited
            error = exc
            time.sleep(min(2 ** attempt, 8))
    return stock_id, f"{type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim/strategy_ohlc_qfq"))
    parser.add_argument("--asset-column", default="stock_id")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--calendar-buffer-days", type=int, default=20)
    parser.add_argument("--minimum-listing-days", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=15.0)
    parser.add_argument("--provider", choices=("tencent", "eastmoney"), default="tencent")
    parser.add_argument(
        "--adjust",
        choices=("", "qfq", "hfq"),
        default="qfq",
        help="AkShare adjustment mode; historical workflow uses hfq",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1 or args.request_timeout <= 0:
        raise ValueError("workers, retries, and request-timeout must be positive")

    predictions = pd.read_parquet(args.predictions)
    required = {args.asset_column, args.date_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {', '.join(sorted(missing))}")
    dates = pd.to_datetime(predictions[args.date_column], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("predictions contain no valid dates")
    start = pd.Timestamp(args.start_date) if args.start_date else dates.min() - pd.Timedelta(
        days=args.calendar_buffer_days
    )
    end = pd.Timestamp(args.end_date) if args.end_date else dates.max() + pd.Timedelta(
        days=args.calendar_buffer_days
    )
    stocks = sorted(
        predictions[args.asset_column].astype(str).str.extract(r"(\d{6})", expand=False).dropna().unique()
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _collect_one,
                stock_id,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                cache_dir=args.cache_dir,
                retries=args.retries,
                request_timeout=args.request_timeout,
                provider=args.provider,
                adjust=args.adjust,
            ): stock_id
            for stock_id in stocks
        }
        completed = 0
        for future in as_completed(futures):
            stock_id, error = future.result()
            completed += 1
            if error:
                failures[stock_id] = error
            if completed % 100 == 0 or completed == len(futures):
                print(json.dumps({
                    "completed": completed, "total": len(futures), "failures": len(failures)
                }))

    parts = []
    for stock_id in stocks:
        path = args.cache_dir / f"{stock_id}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
    if not parts:
        raise ValueError("no OHLC histories were collected")
    ohlc = pd.concat(parts, ignore_index=True)
    panel = build_execution_market_panel(
        ohlc,
        start=dates.min(),
        end=end,
        minimum_listing_days=args.minimum_listing_days,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)
    summary = {
        "predictions": str(args.predictions),
        "output": str(args.output),
        "cache_dir": str(args.cache_dir),
        "provider": args.provider,
        "adjust": args.adjust,
        "requested_stocks": len(stocks),
        "collected_stocks": int(ohlc["stock_id"].nunique()),
        "failures": failures,
        "start": str(start.date()),
        "end": str(end.date()),
        "market_rows": len(panel),
        "observed_rows": int(panel["observed_market"].sum()),
        "eligible_rows": int(panel["eligible_signal"].sum()),
        "ipo_initial_rows": int(panel["ipo_initial_window"].sum()),
        "return_definitions": {
            "open_to_open_return": "adjusted open(t+1) / adjusted open(t) - 1",
            "close_to_close_return": "adjusted close(t+1) / adjusted close(t) - 1",
            "vwap_proxy_return": "adjusted OHLC typical-price proxy, not intraday VWAP",
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
