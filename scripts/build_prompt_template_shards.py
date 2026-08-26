"""Normalize existing news records and split them into deterministic shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def records(source: Path):
    files = sorted(source.rglob("part-*.jsonl")) if source.is_dir() else [source]
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def normalize(row: dict) -> dict:
    if "short_title" not in row:
        row["short_title"] = ""
    if "short_body" not in row:
        full = str(row.get("text_input_2_short", row.get("text_plain", "")) or "")
        old = "平安银行股票未来的涨跌情况如何？请基于下面的文章确定。\n文章："
        marker = "文章："
        pos = full.find(marker)
        row["short_body"] = full[pos + len(marker):] if pos >= 0 else full
    if "masked_short_title" not in row:
        row["masked_short_title"] = ""
    if "masked_short_body" not in row:
        full = str(row.get("text_input_5_masked_short", row.get("text_plain", "")) or "")
        marker = "文章："
        pos = full.find(marker)
        row["masked_short_body"] = full[pos + len(marker):] if pos >= 0 else full
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards", type=int, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    handles = [((args.output / f"part-{i:05d}.jsonl").open("w", encoding="utf-8")) for i in range(args.shards)]
    counts = [0] * args.shards
    try:
        for index, row in enumerate(records(args.source)):
            shard = index % args.shards
            handles[shard].write(json.dumps(normalize(row), ensure_ascii=False) + "\n")
            counts[shard] += 1
    finally:
        for handle in handles:
            handle.close()
    (args.output / "manifest.json").write_text(json.dumps({
        "source": str(args.source), "shards": args.shards,
        "rows": sum(counts), "rows_per_shard": counts,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": sum(counts), "shards": args.shards, "max_shard_rows": max(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()