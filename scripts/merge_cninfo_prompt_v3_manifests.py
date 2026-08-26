"""Validate and merge manifests produced by the 32-way prompt rebuild."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-shards", type=int, default=32)
    args = p.parse_args()
    manifests = []
    for shard in range(args.num_shards):
        path = args.input_dir / f"shard-{shard}.manifest.json"
        if not path.exists():
            raise FileNotFoundError(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("shard_id") != shard or manifest.get("num_shards") != args.num_shards:
            raise ValueError(f"invalid shard metadata: {path}")
        manifests.append(manifest)
    total = sum(int(item["rows"]) for item in manifests)
    source_rows = {int(item["source_rows"]) for item in manifests}
    if len(source_rows) != 1:
        raise ValueError(f"source row counts disagree: {source_rows}")
    if total != next(iter(source_rows)):
        raise ValueError(f"row coverage mismatch: output={total}, source={next(iter(source_rows))}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "input": manifests[0]["input"],
        "source_rows": next(iter(source_rows)),
        "rows": total,
        "num_shards": args.num_shards,
        "prompt_version": "cninfo_prompt_v3_fixed_natural_mask",
        "masking": {"subject": "某公司", "time": "某时间"},
        "short_prompt": "这只股票未来的涨跌情况如何？请基于下面的文章确定。\\n文章：",
        "shards": manifests,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": total, "shards": args.num_shards, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()