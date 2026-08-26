"""Create a deterministic random audit sample from deduplicated CNINFO JSONL."""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    sample: list[dict] = []
    seen = 0
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            seen += 1
            if len(sample) < args.size:
                sample.append(record)
            elif rng.randrange(seen) < args.size:
                sample[rng.randrange(args.size)] = record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        out.write(f"随机审计样本: {len(sample)} 条\n总记录数: {seen}\n随机种子: {args.seed}\n\n")
        for i, record in enumerate(sample, 1):
            text = str(record.get("text", ""))
            flags = {
                "尖括号": bool("<" in text or ">" in text),
                "URL": bool(re.search(r"(?:https?://|www\\.)\\S+", text, re.I)),
                "PDF模板": "当前浏览器不支持在线预览PDF文件" in text,
                "公告下载模板": "公告下载" in text,
                "页码": bool(re.search(r"(?:第\\s*)?\\d+\\s*(?:页|/\\s*\\d+)", text)),
            }
            out.write(f"{'=' * 24} SAMPLE {i:02d} {'=' * 24}\n")
            out.write(f"stock_id: {record.get('stock_id')} | stock_name: {record.get('stock_name')}\n")
            out.write(f"announcement_date: {record.get('announcement_date')}\n")
            out.write(f"title: {record.get('title')}\n")
            out.write(f"source: {record.get('text_source')} | raw_chars: {record.get('raw_text_chars')} | clean_chars: {len(text)}\n")
            out.write(f"quality: low={record.get('low_quality_text')} flags={flags}\n")
            out.write(f"url: {record.get('url')}\n")
            out.write("text_preview:\n")
            out.write(text[:4000].replace("\x00", " "))
            out.write("\n\n")
    print(json.dumps({"input_records": seen, "sample_records": len(sample), "output": str(args.output)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
