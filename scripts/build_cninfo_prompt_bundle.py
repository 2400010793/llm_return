"""Build the complete CNINFO prompt/mask bundle with the frozen v3/v4 rules.

The existing 2018--2026 token assets were built from two directory layouts:
``prompt_v3_fixed_parts`` for short inputs and ``prompt_v4_fixed_long_parts``
for long inputs.  This script deliberately reproduces the record construction
from ``build_cninfo_masked_parts.py`` in one pass, while writing a separate
2010--2026 bundle so the established assets are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_cninfo_prompt_inputs import (
    LONG_PROMPT,
    LONG_TEMPLATE,
    MASKED_LONG_TEMPLATE,
    MASKED_SHORT_TEMPLATE,
    MASKING_VERSION,
    SHORT_PROMPT,
    SHORT_TEMPLATE,
    mask_identity_and_time,
    sha,
)


PROMPT_VERSION = "cninfo_prompt_v3_fixed_point_mask_all_inputs"
VARIANTS = ("plain", "short", "masked_short", "long", "masked_long")
FIXED_LONG_TEMPLATE = LONG_PROMPT + "{title}\n公告正文:{text}"
TEMPLATE_TEXTS = {
    "short": SHORT_TEMPLATE,
    "long": LONG_TEMPLATE,
    "masked_short": MASKED_SHORT_TEMPLATE,
    "masked_long": MASKED_LONG_TEMPLATE,
    "fixed_long": FIXED_LONG_TEMPLATE,
}
TEMPLATE_HASHES = {name: sha(value) for name, value in TEMPLATE_TEXTS.items()}


def build_prompt_record(
    source: dict[str, Any],
    row_index: int,
    *,
    layout: str = "long",
) -> dict[str, Any]:
    """Construct one record using the frozen v3 or v4 field layout."""
    if layout not in ("short", "long"):
        raise ValueError("layout must be 'short' or 'long'")
    text = str(source.get("text_model", source.get("text", "")) or "")
    title = str(source.get("title_clean_final", source.get("title", "")) or "")
    name = str(source.get("stock_name", "") or str(source.get("stock_id", "")))
    stock_id = str(source.get("stock_id", ""))

    short_plain = SHORT_TEMPLATE.format(stock_name=name, text=text)
    long_plain = LONG_TEMPLATE.format(
        stock_name=name,
        stock_id=stock_id,
        title=title,
        text=text,
    )
    masked_text = mask_identity_and_time(text, name, stock_id)
    masked_title = mask_identity_and_time(title, name, stock_id)
    masked_short = MASKED_SHORT_TEMPLATE.format(text=masked_text)
    masked_long = MASKED_LONG_TEMPLATE.format(
        title=masked_title,
        text=masked_text,
    )
    fixed_long = FIXED_LONG_TEMPLATE.format(title=title, text=text)
    fixed_masked_long = FIXED_LONG_TEMPLATE.format(title=masked_title, text=masked_text)

    record = {
        "row_index": row_index,
        "document_id": source.get("document_id"),
        "stock_id_alignment": stock_id,
        "announcement_date_alignment": source.get("announcement_date"),
        "text_plain": text,
        "text_input_2_short": short_plain,
        "text_input_3_long": long_plain,
        "text_input_5_masked_short": masked_short,
        "text_input_6_masked_long": masked_long,
        "text_plain_sha256": sha(text),
        "text_input_2_short_sha256": sha(short_plain),
        "text_input_3_long_sha256": sha(long_plain),
        "text_input_5_masked_short_sha256": sha(masked_short),
        "text_input_6_masked_long_sha256": sha(masked_long),
        "prompt_version": PROMPT_VERSION,
    }
    record.update({
        "short_prompt": SHORT_PROMPT,
        "short_title": title,
        "short_body": text,
        "masked_short_prompt": SHORT_PROMPT,
        "masked_short_title": masked_title,
        "masked_short_body": masked_text,
    })
    if layout == "long":
        record.update({
            "long_prompt": LONG_PROMPT,
            "long_title": title,
            "long_body": text,
            "masked_long_prompt": LONG_PROMPT,
            "masked_long_title": masked_title,
            "masked_long_body": masked_text,
            "text_input_7_fixed_long": fixed_long,
            "text_input_8_fixed_masked_long": fixed_masked_long,
        })
    return record


def _prepare_output_dir(path: Path) -> None:
    """Reset only the dedicated generated prompt directory."""
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"prompt output is not a directory: {path}")
        for child in path.iterdir():
            if child.name.startswith("shard-") or child.name == "manifest.json":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    path.mkdir(parents=True, exist_ok=True)


class PartWriter:
    def __init__(self, output_dir: Path, shard_id: int, part_size: int) -> None:
        self.output_dir = output_dir
        self.shard_id = shard_id
        self.part_size = part_size
        self.part_index = 0
        self.rows_in_part = 0
        self.total_rows = 0
        self.parts: list[dict[str, Any]] = []
        self._final: Path | None = None
        self._temporary: Path | None = None
        self._handle: TextIO | None = None

    def _open(self) -> None:
        shard_dir = self.output_dir / f"shard-{self.shard_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        final = shard_dir / f"part-{self.part_index:05d}.jsonl"
        temporary = final.with_suffix(final.suffix + f".tmp.{os.getpid()}")
        self._final = final
        self._temporary = temporary
        self._handle = temporary.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            self._open()
        assert self._handle is not None
        assert self._final is not None
        assert self._temporary is not None
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.rows_in_part += 1
        self.total_rows += 1
        if self.rows_in_part >= self.part_size:
            self.close_part()

    def close_part(self) -> None:
        if self._handle is None:
            return
        assert self._final is not None
        assert self._temporary is not None
        self._handle.close()
        os.replace(self._temporary, self._final)
        self.parts.append({"path": str(self._final), "rows": self.rows_in_part})
        self.part_index += 1
        self.rows_in_part = 0
        self._final = None
        self._temporary = None
        self._handle = None

    def close(self) -> None:
        self.close_part()


def _write_manifest(
    output_dir: Path,
    input_path: Path,
    source_rows: int,
    writers: list[PartWriter],
    part_size: int,
    num_shards: int,
    row_offset: int,
) -> dict[str, Any]:
    shard_manifests = []
    for writer in writers:
        manifest = {
            "input": str(input_path),
            "source_rows": source_rows,
            "row_index_start": row_offset + 1,
            "row_index_end": row_offset + source_rows,
            "rows": writer.total_rows,
            "part_size": part_size,
            "shard_id": writer.shard_id,
            "num_shards": num_shards,
            "parts": writer.parts,
            "prompt_version": PROMPT_VERSION,
        }
        path = output_dir / f"shard-{writer.shard_id}.manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        shard_manifests.append(manifest)
    root_manifest = {
        "input": str(input_path),
        "source_rows": source_rows,
        "row_index_start": row_offset + 1,
        "row_index_end": row_offset + source_rows,
        "rows": sum(writer.total_rows for writer in writers),
        "part_size": part_size,
        "num_shards": num_shards,
        "prompt_version": PROMPT_VERSION,
        "variants": list(VARIANTS),
        "masking": {
            "version": MASKING_VERSION,
            "subject": "某公司",
            "time": "某时间",
        },
        "template_hashes": TEMPLATE_HASHES,
        "shards": shard_manifests,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--short-output-dir", type=Path, required=True)
    parser.add_argument("--long-output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--part-size", type=int, default=5000)
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument(
        "--row-offset", type=int, default=0,
        help="Global row-index offset for an increment-only input bundle",
    )
    args = parser.parse_args()
    if args.part_size < 1 or args.num_shards < 1:
        raise ValueError("part-size and num-shards must be positive")

    _prepare_output_dir(args.short_output_dir)
    _prepare_output_dir(args.long_output_dir)
    short_writers = [
        PartWriter(args.short_output_dir, shard, args.part_size)
        for shard in range(args.num_shards)
    ]
    long_writers = [
        PartWriter(args.long_output_dir, shard, args.part_size)
        for shard in range(args.num_shards)
    ]
    length_totals: Counter[str] = Counter()
    length_maxima: Counter[str] = Counter()
    source_rows = 0
    try:
        with args.input.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                source_rows += 1
                source_record = json.loads(line)
                source_row_index = source_record.get("row_index", source_rows)
                try:
                    source_row_index = int(source_row_index)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid source row_index at input line {source_rows}") from exc
                expected_row_index = source_rows + args.row_offset
                if source_row_index != expected_row_index:
                    raise ValueError(
                        "prompt input row_index must be contiguous and ordered; "
                        f"line={source_rows} expected={expected_row_index} row_index={source_row_index}"
                    )
                short_record = build_prompt_record(source_record, source_row_index, layout="short")
                long_record = build_prompt_record(source_record, source_row_index, layout="long")
                shard = (source_rows - 1) % args.num_shards
                short_writers[shard].write(short_record)
                long_writers[shard].write(long_record)
                for name in (
                    "text_plain", "text_input_2_short", "text_input_3_long",
                    "text_input_5_masked_short", "text_input_6_masked_long",
                    "text_input_7_fixed_long", "text_input_8_fixed_masked_long",
                ):
                    length = len(str(long_record[name]))
                    length_totals[name] += length
                    length_maxima[name] = max(length_maxima[name], length)
    finally:
        for writer in (*short_writers, *long_writers):
            writer.close()

    short_manifest = _write_manifest(
        args.short_output_dir,
        args.input,
        source_rows,
        short_writers,
        args.part_size,
        args.num_shards,
        args.row_offset,
    )
    long_manifest = _write_manifest(
        args.long_output_dir,
        args.input,
        source_rows,
        long_writers,
        args.part_size,
        args.num_shards,
        args.row_offset,
    )
    bundle = {
        "format_version": "cninfo_prompt_bundle_v1",
        "input": str(args.input),
        "source_rows": source_rows,
        "row_index_start": args.row_offset + 1,
        "row_index_end": args.row_offset + source_rows,
        "short_output_dir": str(args.short_output_dir),
        "long_output_dir": str(args.long_output_dir),
        "short_manifest": short_manifest,
        "long_manifest": long_manifest,
        "prompt_version": PROMPT_VERSION,
        "variants": list(VARIANTS),
        "masking": {
            "version": MASKING_VERSION,
            "subject": "某公司",
            "time": "某时间",
        },
        "template_hashes": TEMPLATE_HASHES,
        "lengths": {
            name: {
                "mean": length_totals[name] / source_rows if source_rows else 0.0,
                "max": length_maxima[name],
            }
            for name in length_totals
        },
        "sha256": hashlib.sha256(
            json.dumps(
                {
                    "input": str(args.input),
                    "source_rows": source_rows,
                    "row_index_start": args.row_offset + 1,
                    "row_index_end": args.row_offset + source_rows,
                    "prompt_version": PROMPT_VERSION,
                    "variants": VARIANTS,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, args.manifest)
    print(json.dumps({"rows": source_rows, "manifest": str(args.manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
