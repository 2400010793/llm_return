"""Build prompt and issuer-masked variants for cleaned single-stock Sina news."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_cninfo_prompt_inputs import mask_identity_and_time


SHORT_TEMPLATE = (
    "任务：根据以下财经新闻，判断目标股票在新闻发布后的下一交易日更可能上涨还是下跌。"
    "只依据新闻中能够获得的信息。\n新闻：{text}"
)
LONG_TEMPLATE = (
    "任务：根据以下财经新闻，判断{stock_name}（股票代码：{stock_id}）在新闻发布后的下一交易日"
    "更可能上涨还是下跌。请识别可能改变投资者预期的信息，包括业绩与现金流、订单与投资、"
    "融资与股权变动、监管与诉讼、经营风险、行业变化及公司治理；忽略广告、栏目导航和无关模板。"
    "不得使用新闻发布后的价格、收益或其他未来信息。\n新闻标题：{title}\n新闻正文：{body}"
)
MASKED_LONG_TEMPLATE = (
    "任务：根据以下财经新闻，判断目标股票在新闻发布后的下一交易日更可能上涨还是下跌。"
    "请识别可能改变投资者预期的信息，包括业绩与现金流、订单与投资、融资与股权变动、"
    "监管与诉讼、经营风险、行业变化及公司治理；忽略广告、栏目导航和无关模板。"
    "不得识别或推测公司名称、股票代码和新闻时间，也不得使用发布后的价格、收益或其他未来信息。"
    "\n新闻标题：{title}\n新闻正文：{body}"
)

VARIANT_COLUMNS = (
    "text_plain",
    "text_prompt_short",
    "text_prompt_long",
    "text_masked_short",
    "text_masked_long",
)


def build_variants(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "article_id", "stock_id", "stock_name", "published_at",
        "publication_date", "title_clean", "body_clean", "text", "text_hash",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"input missing columns: {', '.join(sorted(missing))}")
    if frame["article_id"].isna().any() or frame["article_id"].duplicated().any():
        raise ValueError("article_id must be non-missing and unique")

    rows: list[dict[str, object]] = []
    for record in frame.to_dict("records"):
        title = str(record.get("title_clean", "") or "")
        body = str(record.get("body_clean", "") or "")
        plain = str(record.get("text", "") or "")
        stock_name = str(record.get("stock_name", "") or "")
        stock_id = str(record.get("stock_id", "") or "").zfill(6)
        masked_title = mask_identity_and_time(title, stock_name, stock_id)
        masked_body = mask_identity_and_time(body, stock_name, stock_id)
        masked_plain = mask_identity_and_time(plain, stock_name, stock_id)
        variants = {
            "text_plain": plain,
            "text_prompt_short": SHORT_TEMPLATE.format(text=plain),
            "text_prompt_long": LONG_TEMPLATE.format(
                stock_name=stock_name,
                stock_id=stock_id,
                title=title,
                body=body,
            ),
            "text_masked_short": SHORT_TEMPLATE.format(text=masked_plain),
            "text_masked_long": MASKED_LONG_TEMPLATE.format(
                title=masked_title,
                body=masked_body,
            ),
        }
        rows.append({
            "article_id": record["article_id"],
            "stock_id": stock_id,
            "stock_name": stock_name,
            "publication_date": record["publication_date"],
            "published_at": record["published_at"],
            "text_hash": record["text_hash"],
            **variants,
            **{
                f"{name}_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()
                for name, text in variants.items()
            },
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path(
            "/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cleaned/"
            "sina_single_stock_clean.parquet"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/prompts/"
            "sina_single_stock_prompt_variants.parquet"
        ),
    )
    args = parser.parse_args()
    result = build_variants(pd.read_parquet(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(result),
        "article_ids": int(result["article_id"].nunique()),
        "stocks": int(result["stock_id"].nunique()),
        "variants": {
            name: {
                "mean_chars": float(result[name].str.len().mean()),
                "median_chars": float(result[name].str.len().median()),
                "max_chars": int(result[name].str.len().max()),
                "unique_hashes": int(result[f"{name}_sha256"].nunique()),
            }
            for name in VARIANT_COLUMNS
        },
        "masking": {
            "scope": "stock name/code, company legal names, explicit dates and time expressions",
            "model_visible_placeholders": ["某公司", "某时间"],
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
