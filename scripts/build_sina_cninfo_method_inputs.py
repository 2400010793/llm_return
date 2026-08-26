"""Build sharded Sina prompt/title/body inputs matching the CNINFO method."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_cninfo_prompt_inputs import mask_identity_and_time


SHORT_PROMPT = (
    "任务：根据以下财经新闻，判断目标股票在新闻发布后的下一交易日更可能上涨还是下跌。"
    "只依据新闻中能够获得的信息。\n新闻："
)
LONG_PROMPT = (
    "任务：根据以下财经新闻，判断目标股票在新闻发布后的下一交易日更可能上涨还是下跌。"
    "请识别可能改变投资者预期的信息，包括业绩与现金流、订单与投资、融资与股权变动、"
    "监管与诉讼、经营风险、行业变化及公司治理；忽略广告、栏目导航和无关模板。"
    "不得使用新闻发布后的价格、收益或其他未来信息。\n新闻标题："
)


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def iter_built_records(clean: pd.DataFrame, panel: pd.DataFrame):
    required_clean = {
        "article_id", "stock_id", "stock_name", "title_clean", "body_clean", "text"
    }
    required_panel = {"article_id", "row_index"}
    if missing := required_clean.difference(clean.columns):
        raise ValueError(f"clean input missing columns: {sorted(missing)}")
    if missing := required_panel.difference(panel.columns):
        raise ValueError(f"panel input missing columns: {sorted(missing)}")
    if clean["article_id"].duplicated().any() or panel["article_id"].duplicated().any():
        raise ValueError("article_id must be unique in clean input and panel")
    joined = panel[["article_id", "row_index"]].merge(
        clean, on="article_id", how="left", validate="one_to_one"
    ).sort_values("row_index")
    if joined["text"].isna().any():
        raise ValueError("some panel article_id values are absent from clean input")
    expected = list(range(1, len(joined) + 1))
    if joined["row_index"].astype(int).tolist() != expected:
        raise ValueError("panel row_index must exactly cover 1..N")

    columns = joined.columns.tolist()
    for values in joined.itertuples(index=False, name=None):
        row = dict(zip(columns, values))
        title = str(row.get("title_clean", "") or "")
        body = str(row.get("body_clean", "") or "")
        plain = str(row.get("text", "") or "")
        stock_name = str(row.get("stock_name", "") or "")
        stock_id = str(row.get("stock_id", "") or "").zfill(6)
        masked_title = mask_identity_and_time(title, stock_name, stock_id)
        masked_body = mask_identity_and_time(body, stock_name, stock_id)
        masked_plain = mask_identity_and_time(plain, stock_name, stock_id)
        record = {
            "row_index": int(row["row_index"]),
            "document_id": row["article_id"],
            "article_id": row["article_id"],
            "stock_id_alignment": stock_id,
            "text_plain": plain,
            "short_prompt": SHORT_PROMPT,
            "short_title": title,
            "short_body": body or plain,
            "masked_short_prompt": SHORT_PROMPT,
            "masked_short_title": masked_title,
            "masked_short_body": masked_body or masked_plain,
            "long_prompt": LONG_PROMPT,
            "long_title": title,
            "long_body": body or plain,
            "masked_long_prompt": LONG_PROMPT,
            "masked_long_title": masked_title,
            "masked_long_body": masked_body or masked_plain,
        }
        record.update({
            "text_input_2_short": SHORT_PROMPT + record["short_title"] + record["short_body"],
            "text_input_5_masked_short": (
                SHORT_PROMPT + record["masked_short_title"] + record["masked_short_body"]
            ),
            "text_input_7_fixed_long": (
                LONG_PROMPT + record["long_title"] + "\n新闻正文：" + record["long_body"]
            ),
            "text_input_8_fixed_masked_long": (
                LONG_PROMPT + record["masked_long_title"] + "\n新闻正文："
                + record["masked_long_body"]
            ),
        })
        record["plain_sha256"] = sha256(plain)
        yield record


def build_records(clean: pd.DataFrame, panel: pd.DataFrame) -> list[dict[str, object]]:
    """Materialize records for small callers and unit tests."""
    return list(iter_built_records(clean, panel))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean", type=Path,
        default=Path("/home/gaozh/news_content_quality_20260812/cleaned/sina_single_stock_clean.parquet"),
    )
    parser.add_argument(
        "--panel", type=Path,
        default=Path("/home/gaozh/news_content_quality_20260812/classification/sina_single_stock_classification_panel.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/gaozh/news_content_quality_20260812/cninfo_method_v1/inputs"),
    )
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    records = iter_built_records(
        pd.read_parquet(args.clean), pd.read_parquet(args.panel)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = [0] * args.shards
    handles = []
    for shard in range(args.shards):
        directory = args.output_dir / f"shard-{shard}"
        directory.mkdir(parents=True, exist_ok=True)
        handles.append((directory / "part-00000.jsonl").open("w", encoding="utf-8"))
    try:
        for row in records:
            shard = (int(row["row_index"]) - 1) % args.shards
            handles[shard].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[shard] += 1
    finally:
        for handle in handles:
            handle.close()
    summary = {
        "rows": sum(counts), "shards": args.shards, "shard_rows": counts,
        "row_index": {
            "minimum": 1, "maximum": sum(counts), "unique": sum(counts)
        },
        "variants": ["plain", "short", "masked_short", "long", "masked_long"],
        "paired_prompts": {
            "short_equals_masked_short": True,
            "long_equals_masked_long": True,
        },
        "pooled_representations": [
            "prompt_mean", "title_mean", "body_mean", "title_body_mean",
            "full_mean", "title_body_concat", "title_body_full_concat",
        ],
        "prompt_token_order_retained": True,
        "short_prompt": SHORT_PROMPT, "long_prompt": LONG_PROMPT,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
