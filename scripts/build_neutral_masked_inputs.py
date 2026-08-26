"""Build deterministic full-history masked-short inputs for Sina and CNINFO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_cninfo_prompt_inputs import mask_identity_and_time


BASE_PROMPT = "分析股票风险"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def make_record(
    *, row_index: int, document_id: str, stock_id: str, stock_name: str,
    date_value: Any, title: str, body: str, source: str,
) -> dict[str, Any]:
    sid = normalize(stock_id).strip().zfill(6)
    name = normalize(stock_name).strip() or sid
    title = normalize(title)
    body = normalize(body)
    masked_title = mask_identity_and_time(title, name, sid)
    masked_body = mask_identity_and_time(body, name, sid)
    masked_full = BASE_PROMPT + masked_title + masked_body
    plain = title + body
    return {
        "row_index": int(row_index),
        "document_id": normalize(document_id),
        "stock_id_alignment": sid,
        "stock_name_alignment": name,
        "announcement_date_alignment": normalize(date_value),
        "source_dataset": source,
        "text_plain": plain,
        "plain_sha256": digest(plain),
        "masked_short_prompt": BASE_PROMPT,
        "masked_short_title": masked_title,
        "masked_short_body": masked_body,
        "text_input_5_masked_short": masked_full,
        "text_input_5_masked_short_sha256": digest(masked_full),
    }


class ContiguousShardWriter:
    """Keep only the current contiguous shard open on Lustre."""

    def __init__(self, output: Path):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.current_shard: int | None = None
        self.handle = None
        self.temporary: Path | None = None
        self.final: Path | None = None
        self.part_index = 0
        self.bytes_in_part = 0
        self.max_part_bytes = 8 * 1024 * 1024
        self.local_root = Path(os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()) / f"neutral-input-{os.getpid()}"
        self.local_root.mkdir(parents=True, exist_ok=True)

    def write(self, shard: int, record: dict[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if self.current_shard != shard:
            self.close()
            self.part_index = 0
            directory = self.output / f"shard-{shard}"
            directory.mkdir(parents=True, exist_ok=True)
            self.final = directory / f"part-{self.part_index:05d}.jsonl"
            self.temporary = self.local_root / f"shard-{shard}-part-{self.part_index:05d}.jsonl"
            self.handle = self.temporary.open("wb")
            self.current_shard = shard
            self.bytes_in_part = 0
        elif self.bytes_in_part and self.bytes_in_part + len(payload) > self.max_part_bytes:
            self.close()
            self.part_index += 1
            directory = self.output / f"shard-{shard}"
            self.final = directory / f"part-{self.part_index:05d}.jsonl"
            self.temporary = self.local_root / f"shard-{shard}-part-{self.part_index:05d}.jsonl"
            self.handle = self.temporary.open("wb")
            self.current_shard = shard
            self.bytes_in_part = 0
        self.handle.write(payload)
        self.bytes_in_part += len(payload)

    def close(self) -> None:
        if self.handle is None:
            return
        self.handle.close()
        assert self.temporary is not None and self.final is not None
        shutil.copyfile(self.temporary, self.final)
        self.temporary.unlink(missing_ok=True)
        self.handle = None
        self.temporary = None
        self.final = None
        self.current_shard = None
        self.bytes_in_part = 0

    def cleanup(self) -> None:
        self.close()
        shutil.rmtree(self.local_root, ignore_errors=True)


def build_sina(clean_path: Path, panel_path: Path, output: Path, shards: int) -> dict[str, Any]:
    panel = pd.read_parquet(panel_path, columns=["row_index", "article_id", "stock_id"])
    if len(panel) == 0 or panel["article_id"].duplicated().any():
        raise ValueError("Sina panel must have unique article_id")
    row_by_id = dict(zip(panel["article_id"].astype(str), panel["row_index"].astype(int)))
    stock_by_id = dict(zip(panel["article_id"].astype(str), panel["stock_id"].astype(str)))
    writer = ContiguousShardWriter(output)
    rows_per_shard = (len(panel) + shards - 1) // shards
    count = 0
    try:
        parquet = pq.ParquetFile(clean_path)
        source_rows = 0
        columns = [
            "article_id", "stock_id", "stock_name", "published_at",
            "title_clean", "body_clean", "text",
        ]
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
          for row in batch.to_pylist():
            source_rows += 1
            article_id = normalize(row.get("article_id"))
            row_index = row_by_id.get(article_id)
            if row_index is None:
                raise ValueError(f"Sina article missing from panel: {article_id}")
            expected = count + 1
            if row_index != expected:
                raise ValueError(f"Sina row order mismatch at {expected}: got {row_index}")
            title = normalize(row.get("title_clean"))
            body = normalize(row.get("body_clean")) or normalize(row.get("text"))
            record = make_record(
                row_index=row_index, document_id=article_id,
                stock_id=stock_by_id[article_id], stock_name=normalize(row.get("stock_name")),
                date_value=row.get("published_at"), title=title, body=body, source="sina",
            )
            writer.write(min((row_index - 1) // rows_per_shard, shards - 1), record)
            count += 1
        if source_rows != len(panel):
            raise ValueError(f"Sina clean/panel rows differ: {source_rows} != {len(panel)}")
    finally:
        writer.cleanup()
    if count != len(panel):
        raise ValueError(f"Sina rows written {count} != panel {len(panel)}")
    return {"dataset": "sina", "rows": count, "shards": shards}


def build_cninfo(source_path: Path, panel_path: Path, output: Path, shards: int) -> dict[str, Any]:
    panel = pd.read_parquet(panel_path, columns=[
        "row_index", "document_id", "stock_id", "stock_name", "announcement_date", "url",
    ])
    if len(panel) == 0 or panel["url"].duplicated().any():
        raise ValueError("CNINFO panel must have unique URL")
    by_url = {
        str(row.url): {
            "row_index": int(row.row_index), "document_id": str(row.document_id),
            "stock_id": str(row.stock_id), "stock_name": normalize(row.stock_name),
            "date": row.announcement_date,
        }
        for row in panel.itertuples(index=False)
    }
    writer = ContiguousShardWriter(output)
    rows_per_shard = (len(panel) + shards - 1) // shards
    count = 0
    seen = set()
    try:
        with source_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                match = by_url.get(normalize(item.get("url")))
                if match is None:
                    continue
                row_index = match["row_index"]
                if row_index in seen:
                    raise ValueError(f"duplicate CNINFO row_index {row_index} at source line {line_number}")
                expected = count + 1
                if row_index != expected:
                    raise ValueError(f"CNINFO row order mismatch at {expected}: got {row_index}")
                seen.add(row_index)
                title = normalize(item.get("title_clean_final", item.get("title")))
                body = normalize(item.get("text_model", item.get("text")))
                record = make_record(
                    row_index=row_index, document_id=match["document_id"],
                    stock_id=match["stock_id"], stock_name=match["stock_name"],
                    date_value=match["date"], title=title, body=body, source="cninfo",
                )
                writer.write(min((row_index - 1) // rows_per_shard, shards - 1), record)
                count += 1
    finally:
        writer.cleanup()
    if count != len(panel) or seen != set(range(1, len(panel) + 1)):
        raise ValueError(f"CNINFO rows written {count} != panel {len(panel)}")
    return {"dataset": "cninfo", "rows": count, "shards": shards}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sina", "cninfo"), required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=256)
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    if args.dataset == "sina":
        summary = build_sina(args.clean, args.panel, args.output, args.shards)
    else:
        summary = build_cninfo(args.clean, args.panel, args.output, args.shards)
    summary.update({"clean": str(args.clean), "panel": str(args.panel), "base_prompt": BASE_PROMPT})
    (args.output / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
