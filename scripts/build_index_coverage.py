from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "data/stock_universe_paper_1000.csv"
OUT_UNIVERSE = ROOT / "data/stock_universe_index_coverage.csv"
SOURCE_DIR = ROOT / "data/source_stock_lists"
REPORT = ROOT / "reports/index_coverage_current.json"

INDEXES = {
    "CSI300": ("000300", "沪深300"),
    "CSI500": ("000905", "中证500"),
}


def normalize_code(value: object) -> str:
    text = str(value).strip()
    return text.zfill(6)


def fetch_index(code: str, name: str) -> pd.DataFrame:
    frame = ak.index_stock_cons_csindex(symbol=code).copy()
    frame["stock_id"] = frame["成分券代码"].map(normalize_code)
    frame["stock_name"] = frame["成分券名称"].astype(str).str.strip()
    frame["exchange"] = frame["交易所"].map(
        lambda x: "SH" if "上海" in str(x) else "SZ" if "深圳" in str(x) else ""
    )
    frame["index_key"] = name
    frame["index_code"] = code
    frame["as_of_date"] = frame["日期"].astype(str)
    return frame[["stock_id", "stock_name", "exchange", "index_key", "index_code", "as_of_date"]]


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    index_frames = []
    for key, (code, name) in INDEXES.items():
        frame = fetch_index(code, name)
        frame.to_csv(SOURCE_DIR / f"{key.lower()}_constituents.csv", index=False, encoding="utf-8-sig")
        index_frames.append(frame)
    constituents = pd.concat(index_frames, ignore_index=True)

    universe = pd.read_csv(UNIVERSE, dtype=str).fillna("")
    universe["stock_id"] = universe["stock_id"].map(normalize_code)
    universe_ids = set(universe["stock_id"])

    memberships = (
        constituents.groupby("stock_id")
        .agg(
            index_membership=("index_key", lambda values: ";".join(sorted(set(values)))),
            index_codes=("index_code", lambda values: ";".join(sorted(set(values)))),
            index_as_of=("as_of_date", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
    )
    enriched = universe.merge(memberships, on="stock_id", how="left")
    for column in ["index_membership", "index_codes", "index_as_of"]:
        enriched[column] = enriched[column].fillna("")

    missing = constituents[~constituents["stock_id"].isin(universe_ids)].copy()
    missing = missing.drop_duplicates("stock_id")
    missing["industry"] = ""
    missing["xueqiu_symbol"] = missing.apply(lambda row: f"{row['exchange']}{row['stock_id']}", axis=1)
    missing["active"] = "1"
    missing["index_membership"] = missing["stock_id"].map(
        constituents.groupby("stock_id")["index_key"].apply(lambda values: ";".join(sorted(set(values))))
    )
    missing["index_codes"] = missing["stock_id"].map(
        constituents.groupby("stock_id")["index_code"].apply(lambda values: ";".join(sorted(set(values))))
    )
    missing["index_as_of"] = missing["stock_id"].map(
        constituents.groupby("stock_id")["as_of_date"].apply(lambda values: ";".join(sorted(set(values))))
    )
    missing = missing[["stock_id", "stock_name", "exchange", "industry", "xueqiu_symbol", "active", "index_membership", "index_codes", "index_as_of"]]
    expanded = pd.concat([enriched, missing], ignore_index=True).drop_duplicates("stock_id")
    expanded.to_csv(OUT_UNIVERSE, index=False, encoding="utf-8-sig")

    coverage = {}
    for key, (_, name) in INDEXES.items():
        members = set(constituents.loc[constituents["index_key"] == name, "stock_id"])
        present = members & universe_ids
        absent = sorted(members - universe_ids)
        coverage[key] = {
            "index_name": name,
            "index_code": next(code for code, n in INDEXES.values() if n == name),
            "as_of_date": sorted(set(constituents.loc[constituents["index_key"] == name, "as_of_date"])),
            "constituent_count": len(members),
            "present_in_original_universe": len(present),
            "missing_from_original_universe": len(absent),
            "coverage_rate": len(present) / len(members) if members else 0.0,
            "missing_stock_ids": absent,
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "AkShare index_stock_cons_csindex; CSI official constituent endpoint",
        "source_retrieved_on": str(date.today()),
        "original_universe": str(UNIVERSE.relative_to(ROOT)),
        "expanded_universe": str(OUT_UNIVERSE.relative_to(ROOT)),
        "original_universe_count": len(universe),
        "expanded_universe_count": len(expanded),
        "new_unique_constituents": len(missing),
        "coverage": coverage,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "expanded": str(OUT_UNIVERSE), "new": len(missing), "coverage": coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
