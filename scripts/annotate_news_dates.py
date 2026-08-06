"""Add conservative day-level publication-date labels to collected JSON records.

This never treats ``collected_at`` as publication time.  Exact minutes are kept
only when exposed by the source; otherwise the record is marked ``date`` or
``unknown`` so downstream daily return labeling can filter explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


CHINA_TZ = timezone(timedelta(hours=8))
FULL_DATE = re.compile(r"(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})(?:日)?(?:\s+([0-9]{1,2}):([0-9]{2}))?")
MONTH_DAY_TIME = re.compile(r"(\d{1,2})[-月/](\d{1,2})(?:日)?\s+([0-9]{1,2}):([0-9]{2})")


def parse_record(record: dict) -> tuple[str | None, str, str | None]:
    """Return ISO timestamp, precision, and source without inventing time."""
    if record.get("announcement_date"):
        value = str(record["announcement_date"])
        match = FULL_DATE.search(value)
        if match:
            return f"{int(match[1]):04d}-{int(match[2]):02d}-{int(match[3]):02d}T00:00:00+08:00", "date", "announcement_date"

    for field in ("published_at_display", "published_at"):
        value = str(record.get(field) or "")
        match = FULL_DATE.search(value)
        if match:
            hour = int(match[4] or 0)
            minute = int(match[5] or 0)
            precision = "minute" if match[4] is not None else "date"
            return f"{int(match[1]):04d}-{int(match[2]):02d}-{int(match[3]):02d}T{hour:02d}:{minute:02d}:00+08:00", precision, field

    updated = MONTH_DAY_TIME.search(str(record.get("last_updated") or ""))
    if updated:
        year = str(record.get("collected_at") or "")[:4]
        if year.isdigit():
            return f"{int(year):04d}-{int(updated[1]):02d}-{int(updated[2]):02d}T{int(updated[3]):02d}:{int(updated[4]):02d}:00+08:00", "minute", "last_updated"

    body = str(record.get("body") or record.get("visible_text") or "")
    match = FULL_DATE.search(body)
    if match:
        hour = int(match[4] or 0)
        minute = int(match[5] or 0)
        precision = "minute" if match[4] is not None else "date"
        return f"{int(match[1]):04d}-{int(match[2]):02d}-{int(match[3]):02d}T{hour:02d}:{minute:02d}:00+08:00", precision, "body"
    return None, "unknown", None


def annotate(record: dict) -> dict:
    timestamp, precision, source = parse_record(record)
    item = {**record, "published_at": timestamp, "published_at_precision": precision, "published_at_source": source, "timezone": "Asia/Shanghai"}
    item["news_date"] = timestamp[:10] if timestamp else None
    item["after_market_close"] = bool(timestamp and precision == "minute" and timestamp[11:16] >= "15:00")
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="collector JSON files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for filename in args.inputs:
        records.extend(json.loads(Path(filename).read_text(encoding="utf-8")).get("records", []))
    annotated = [annotate(record) for record in records]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"records": annotated}, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {"records": len(annotated), "date": sum(r["published_at_precision"] == "date" for r in annotated), "minute": sum(r["published_at_precision"] == "minute" for r in annotated), "unknown": sum(r["published_at_precision"] == "unknown" for r in annotated), "output": str(destination)}
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()