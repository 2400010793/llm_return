"""Freeze the exact CNINFO intersection across all completed Qwen prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROMPTS = ("future_return", "return", "loss", "excess_return", "profit")
VARIANTS = ("short", "masked_short")


def shard_ids(root: Path, prompt: str, variant: str) -> set[int]:
    return {
        int(marker.parent.name.split("-")[1])
        for marker in (root / prompt / variant).glob("shard-*/COMPLETED")
    }


def read_row_indices(shard: Path) -> list[int]:
    rows = []
    with (shard / "metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(int(json.loads(line)["row_index"]))
    return rows


def digest_ints(values: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, values)).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sets = {
        (prompt, variant): shard_ids(args.root, prompt, variant)
        for prompt in PROMPTS for variant in VARIANTS
    }
    common_shards = sorted(set.intersection(*sets.values()))
    if not common_shards:
        raise ValueError("no shard is complete for all prompt/variant combinations")

    all_rows: list[int] = []
    shard_rows = []
    reference = ("return", "masked_short")
    for shard_id in common_shards:
        ref_dir = args.root / reference[0] / reference[1] / f"shard-{shard_id}"
        rows = read_row_indices(ref_dir)
        if not rows or len(rows) != len(set(rows)):
            raise ValueError(f"invalid reference metadata in shard {shard_id}")
        for prompt in PROMPTS:
            for variant in VARIANTS:
                current = read_row_indices(
                    args.root / prompt / variant / f"shard-{shard_id}"
                )
                if current != rows:
                    raise ValueError(
                        f"row order mismatch: {prompt}/{variant}/shard-{shard_id}"
                    )
        shard_rows.append({
            "shard_id": shard_id,
            "rows": len(rows),
            "row_index_sha256": digest_ints(rows),
            "row_index_min": min(rows),
            "row_index_max": max(rows),
        })
        all_rows.extend(rows)
    if len(all_rows) != len(set(all_rows)):
        raise ValueError("row_index is duplicated across common shards")

    panel = pd.read_parquet(args.panel)
    if "row_index" not in panel:
        raise ValueError("panel is missing row_index")
    qwen_rows = pd.DataFrame({
        "qwen_row_index": all_rows,
        "row_index": [value + 1 for value in all_rows],
        "intersection_position": range(len(all_rows)),
    })
    selected = qwen_rows.merge(panel, on="row_index", how="left", validate="one_to_one")
    required = ("stock_id", "entry_date")
    if selected[list(required)].isna().any(axis=1).any():
        raise ValueError("one or more Qwen rows do not map to the CNINFO panel")
    selected["entry_date"] = pd.to_datetime(selected["entry_date"], errors="coerce")
    selected["year"] = selected["entry_date"].dt.year
    years = {
        str(int(year)): int(count)
        for year, count in selected["year"].value_counts().sort_index().items()
        if pd.notna(year)
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output_dir / "intersection_panel.parquet", index=False)
    qwen_rows.to_parquet(args.output_dir / "intersection_metadata.parquet", index=False)
    manifest = {
        "format_version": "qwen_cninfo_strict_intersection_v1",
        "root": str(args.root),
        "panel": str(args.panel),
        "prompts": list(PROMPTS),
        "variants": list(VARIANTS),
        "required_representations": 10,
        "common_shards": common_shards,
        "common_shard_count": len(common_shards),
        "rows": len(all_rows),
        "row_index_sha256": digest_ints(all_rows),
        "row_index_mapping": "panel_row_index = qwen_row_index + 1",
        "year_rows": years,
        "completed_shards_per_representation": {
            f"{prompt}/{variant}": len(values)
            for (prompt, variant), values in sets.items()
        },
        "shards": shard_rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "common_shards": len(common_shards), "rows": len(all_rows), "years": years,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
