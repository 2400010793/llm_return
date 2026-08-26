"""Download a public CSI Index constituent file and normalize it to the project stock schema."""
from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_URL = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000905cons.xls"
OUTPUT_FIELDS = ["stock_id", "stock_name", "exchange", "industry", "ticker", "active"]


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; llm-return/0.1)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for column in frame.columns:
        text = str(column).strip().lower()
        if any(candidate in text for candidate in candidates):
            return column
    raise ValueError(f"could not find one of {candidates}; columns={list(frame.columns)!r}")


def normalize(payload: bytes) -> list[dict[str, str | int]]:
    frame = pd.read_excel(io.BytesIO(payload), engine="xlrd")
    code_column = find_column(frame, ("成份券代码", "成分券代码", "constituent code"))
    name_column = find_column(frame, ("成份券名称", "成分券名称", "constituent name"))
    rows: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        raw_code = str(row[code_column]).strip()
        if raw_code.endswith(".0"):
            raw_code = raw_code[:-2]
        code = raw_code.zfill(6)
        name = str(row[name_column]).strip()
        if len(code) != 6 or not code.isdigit() or code in seen or not name or name == "nan":
            continue
        seen.add(code)
        exchange = "SH" if code.startswith(("5", "6", "688", "689")) else "SZ"
        rows.append(
            {
                "stock_id": code,
                "stock_name": name,
                "exchange": exchange,
                "industry": "",
                "ticker": f"{exchange}{code}",
                "active": 1,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-input", default="")
    parser.add_argument("--raw-output", default="data/interim/csindex_000905cons.xls")
    parser.add_argument("--output", default="data/stock_universe_csi500_current.csv")
    parser.add_argument("--limit", type=int, default=0, help="仅输出前 N 个成分股，用于小规模试采集")
    args = parser.parse_args()

    payload = Path(args.raw_input).read_bytes() if args.raw_input else download(args.url)
    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(payload)
    rows = normalize(payload)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("the public CSI constituent file contained no stock rows")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print({"index": "000905", "stocks": len(rows), "raw_output": str(raw_path), "output": str(output)})


if __name__ == "__main__":
    main()
