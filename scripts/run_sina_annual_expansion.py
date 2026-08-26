"""Run independent historical Sina expansions for each year.

The root set is split by the year encoded in each root URL.  Each year gets
its own queue and output files, while the browser selector may still follow
cross-year links for discovery.  A small number of year jobs run concurrently;
within each job the browser worker count is separately bounded.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

YEAR_IN_PATH = re.compile(r"/(20\d{2})(?:\d{4}/|-\d{2}-\d{2}/|/)")


def root_year(url: str) -> int | None:
    match = YEAR_IN_PATH.search(urlparse(url).path)
    return int(match.group(1)) if match else None


def split_roots(root_file: Path, root_dir: Path) -> dict[int, Path]:
    grouped: dict[int, list[str]] = {}
    for raw in root_file.read_text(encoding="utf-8").splitlines():
        url = raw.strip()
        if not url or url.startswith("#"):
            continue
        year = root_year(url)
        if year is not None:
            grouped.setdefault(year, []).append(url)
    root_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for year, urls in sorted(grouped.items()):
        path = root_dir / f"roots_{year}.txt"
        path.write_text("\n".join(dict.fromkeys(urls)) + "\n", encoding="utf-8")
        paths[year] = path
    return paths


def run_year(year: int, root_path: Path, args: argparse.Namespace) -> dict:
    prefix = args.output_dir / f"sina_{year}_depth{args.max_depth}_w{args.page_workers}"
    command = [
        sys.executable,
        str(args.selector),
        "--root-file", str(root_path),
        "--years", str(year),
        "--per-year", str(args.quotas.get(str(year), args.per_year)),
        "--max-per-stock", str(args.max_per_stock),
        "--max-pages", str(args.max_pages),
        "--max-depth", str(args.max_depth),
        "--workers", str(args.page_workers),
        "--pause-seconds", str(args.pause_seconds),
        "--render-wait-ms", str(args.render_wait_ms),
        "--output", str(prefix.with_suffix(".json")),
        "--stream-output", str(prefix.with_suffix(".records.jsonl")),
        "--catalog-output", str(prefix.with_suffix(".csv")),
    ]
    if args.allow_multi_stock:
        command.append("--allow-multi-stock")
    result = subprocess.run(command, cwd=args.project_dir, capture_output=True, text=True, encoding="utf-8", errors="replace")
    summary = {
        "year": year,
        "returncode": result.returncode,
        "root_file": str(root_path),
        "output": str(prefix.with_suffix(".json")),
        "stream_output": str(prefix.with_suffix(".records.jsonl")),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }
    if result.returncode == 0:
        try:
            payload = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
            summary["visited"] = payload.get("visited", 0)
            summary["selected"] = len(payload.get("selected", []))
        except (OSError, json.JSONDecodeError):
            summary["selected"] = None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="按年份独立扩展新浪历史新闻 seed")
    parser.add_argument("--root-file", required=True)
    parser.add_argument("--selector", default="scripts/select_sina_historical_seeds_browser.py")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--output-dir", default="data/interim/annual_expansions")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--year-workers", type=int, default=4, help="同时运行多少个年份任务")
    parser.add_argument("--page-workers", type=int, default=2, help="每个年份任务使用多少浏览器页面")
    parser.add_argument("--per-year", type=int, default=100)
    parser.add_argument("--quota-file", help="JSON object mapping year to per-year target")
    parser.add_argument("--max-per-stock", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--render-wait-ms", type=int, default=300)
    parser.add_argument("--allow-multi-stock", action="store_true")
    args = parser.parse_args()
    args.selector = Path(args.selector).resolve()
    args.project_dir = Path(args.project_dir).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    args.quotas = {}
    if args.quota_file:
        args.quotas = json.loads(Path(args.quota_file).resolve().read_text(encoding="utf-8"))
    root_paths = split_roots(Path(args.root_file).resolve(), args.output_dir / "roots")
    jobs = {year: path for year, path in root_paths.items() if args.start_year <= year <= args.end_year}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.year_workers) as pool:
        futures = {pool.submit(run_year, year, path, args): year for year, path in jobs.items()}
        for future in as_completed(futures):
            results.append(future.result())
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    results.sort(key=lambda item: item["year"])
    manifest = args.output_dir / "annual_expansion_manifest.json"
    manifest.write_text(json.dumps({"parameters": vars(args) | {"selector": str(args.selector), "project_dir": str(args.project_dir), "output_dir": str(args.output_dir)}, "results": results}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"years_started": len(jobs), "successful": sum(x["returncode"] == 0 for x in results), "manifest": str(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
