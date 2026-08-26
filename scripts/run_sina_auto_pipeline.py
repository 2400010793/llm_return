"""Run resumable Sina crawl -> global dedup -> depth-5 expansion."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

from crawl_sina_bfs_fast import crawl


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def year_of(record: dict) -> str | None:
    value = str(record.get("published_at") or "")
    match = re.search(r"(20\d{2})", value)
    if match:
        return match.group(1)
    match = re.search(r"/(20\d{2})\d{4}/", str(record.get("url") or ""))
    return match.group(1) if match else None


def dedup_records(records: list[dict], output: Path) -> list[dict]:
    seen_urls: set[str] = set()
    seen_bodies: set[str] = set()
    kept: list[dict] = []
    for record in records:
        if year_of(record) == "2000":
            continue
        url = record.get("url") or ""
        body_key = hashlib.sha256(str(record.get("body") or "").strip().encode("utf-8")).hexdigest()
        if url in seen_urls or body_key in seen_bodies:
            continue
        seen_urls.add(url)
        seen_bodies.add(body_key)
        kept.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"records": kept, "count": len(kept), "excluded_year": "2000"}, ensure_ascii=False, indent=2), encoding="utf-8")
    return kept


def run_stage(name: str, roots: list[str] | None, root_file: Path | None, out_dir: Path, depth: int, args) -> dict:
    output = out_dir / f"{name}.json"
    namespace = SimpleNamespace(
        output=str(output), raw_dir=str(out_dir / "raw"), state_file=str(out_dir / f"{name}.state.json"),
        resume=True, root=roots or [], max_pages=None, max_depth=depth, max_seconds=args.max_seconds,
        workers=args.workers, pause_seconds=args.pause_seconds, timeout=args.timeout, min_body_chars=args.min_body_chars,
        max_body_chars=args.max_body_chars, stock_catalog=args.stock_catalog,
        exactly_one_stock=False, paper_stock_matching=True, user_agent=args.user_agent,
    )
    if root_file:
        namespace.root = [line.strip() for line in root_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    return crawl(namespace)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sina resumable crawl, deduplication, and depth-5 expansion")
    parser.add_argument("--root-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stock-catalog", required=True)
    parser.add_argument("--workers", type=int, default=20, help="并发请求数，范围1-50")
    parser.add_argument("--max-seconds", type=float, default=None, help="运行时间上限；省略表示两阶段均运行至 frontier 耗尽")
    parser.add_argument("--pause-seconds", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--min-body-chars", type=int, default=120)
    parser.add_argument("--max-body-chars", type=int, default=12000)
    parser.add_argument("--user-agent", default="llm-return-research/0.1 (academic prototype)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 50:
        parser.error("workers 应为 1-50")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1 = run_stage("stage1_depth3_all", None, Path(args.root_file), out_dir / "stage1", 3, args)
    records = read_records(out_dir / "stage1" / "stage1_depth3_all.records.jsonl")
    cleaned = dedup_records(records, out_dir / "dedup" / "articles_dedup_no2000.json")
    roots = out_dir / "dedup" / "article_roots.txt"
    roots.write_text("\n".join(record["url"] for record in cleaned if record.get("url")) + "\n", encoding="utf-8")
    stage2 = run_stage("stage2_depth5", None, roots, out_dir / "stage2", 5, args)
    final_records = read_records(out_dir / "stage2" / "stage2_depth5.records.jsonl")
    dedup_records(final_records, out_dir / "dedup" / "all_articles_after_depth5.json")
    print(json.dumps({"stage1_visited": stage1["visited"], "stage1_records": len(records), "dedup_roots": len(cleaned), "stage2_visited": stage2["visited"], "stage2_records": len(final_records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
