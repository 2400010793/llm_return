"""Build prompt inputs, including anonymized variants 5 and 6."""
from __future__ import annotations
import argparse, hashlib, json, os, re
from pathlib import Path

# Keep the visible short prompt identical across raw and masked variants.  The
# issuer identity belongs only to the body/title input, not to the instruction.
SHORT_PROMPT = "这只股票未来的涨跌情况如何？请基于下面的文章确定。\n文章："
SHORT_TEMPLATE = SHORT_PROMPT + "{text}"
LONG_PROMPT = (
    "请基于下面的公告文章，判断这只股票未来短期内的收益方向更可能上涨还是下跌。"
    "请重点识别可能改变投资者预期的信息，包括业绩与现金流、重大合同与投资、融资与股权变动、诉讼与监管处罚、"
    "经营风险、公司治理及其他重大事项；忽略公告格式、法律模板和与收益无关的礼节性内容。"
    "只允许依据公告发布时能够获得的信息，不得使用公告发布后的股价、收益或其他未来信息。\n"
    "公告标题："
    "公告正文："
)
LONG_TEMPLATE = (
    "任务：请基于下面的公告文章，分析{stock_name}（股票代码：{stock_id}）未来短期内的股票收益方向更可能上涨还是下跌。"
    "请重点识别可能改变投资者预期的信息，包括业绩与现金流、重大合同与投资、融资与股权变动、诉讼与监管处罚、"
    "经营风险、公司治理及其他重大事项；忽略公告格式、法律模板和与收益无关的礼节性内容。"
    "只允许依据公告发布时能够获得的信息，不得使用公告发布后的股价、收益或其他未来信息。\n"
    "公告标题：{title}\n公告正文：{text}"
)
# Masking changes title/body only; the instruction remains identical.
MASKED_SHORT_TEMPLATE = SHORT_PROMPT + "{text}"
MASKED_LONG_TEMPLATE = (
    "任务：请仅基于下面的公告文章，判断目标股票未来短期内的股票收益方向更可能上涨还是下跌。"
    "请重点识别可能改变投资者预期的信息，包括业绩与现金流、重大合同与投资、融资与股权变动、诉讼与监管处罚、"
    "经营风险、公司治理及其他重大事项；忽略公告格式、法律模板和与收益无关的礼节性内容。"
    "不得识别或推测公司名称、股票代码和公告时间，也不得使用公告发布后的股价、收益或其他未来信息。\n"
    "公告标题：{title}\n公告正文：{text}"
)
MASKED_LONG_FIXED_TEMPLATE = LONG_PROMPT + "{title}\n公告正文：{text}"
# Natural-language placeholders avoid unknown/split symbolic tokens in the
# tokenizer while preserving the fact that identity/time was removed.
SUBJECT_MASK = "某公司"
TIME_MASK = "某时间"
MASKING_VERSION = "cninfo_mask_v3_fixed_point"
MAX_MASK_PASSES = 1000
COMPANY_NAME = re.compile(r"[\u4e00-\u9fffA-Za-z0-9（）()·]{2,50}(?:股份有限公司|有限责任公司|有限公司)")
DATE_PATTERNS = (
    re.compile(r"[〇○零一二三四五六七八九十百]{4}\s*年\s*(?:[〇○零一二三四五六七八九十百]{1,3}\s*月(?:\s*[〇○零一二三四五六七八九十百]{1,3}\s*日?)?)?"),
    re.compile(r"(?<!\d)(?:19|20)\d{2}\s*[年./-]\s*\d{1,2}\s*[月./-]\s*\d{1,2}\s*日?"),
    re.compile(r"(?<!\d)(?:19|20)\d{2}\s*年\s*\d{1,2}\s*月"),
    re.compile(r"(?<!\d)(?:19|20)\d{2}\s*(?:年|年度|财年)"),
    re.compile(r"(?<!\d)\d{1,2}\s*月\s*\d{1,2}\s*日"),
    re.compile(r"(?<!\d)\d{1,2}\s*[时:]\s*\d{2}\s*分?"),
    re.compile(r"第[一二三四1-4]季度|上半年|下半年|年初|年末|年底"),
)


def _mask_identity_and_time_once(value: str, stock_name: str, stock_id: str) -> str:
    """Apply one masking pass; later replacements may expose new matches."""
    text = value
    names = {stock_name, re.sub(r"\s+", "", stock_name)}
    for name in sorted((name for name in names if len(name) >= 2), key=len, reverse=True):
        text = text.replace(name, SUBJECT_MASK)
    if stock_id:
        text = re.sub(rf"(?<!\d){re.escape(stock_id)}(?!\d)", SUBJECT_MASK, text)
    text = COMPANY_NAME.sub(SUBJECT_MASK, text)
    for pattern in DATE_PATTERNS:
        text = pattern.sub(TIME_MASK, text)
    return text


def mask_identity_and_time(value: str, stock_name: str, stock_id: str) -> str:
    """Mask identity/time patterns to a fixed point.

    OCR text frequently concatenates dates and legal names without separators.
    Replacing one match can therefore expose the next match to a look-behind or
    overlapping suffix. Iterate until another full pass makes no change.
    """
    text = value
    for _ in range(MAX_MASK_PASSES):
        masked = _mask_identity_and_time_once(text, stock_name, stock_id)
        if masked == text:
            return masked
        text = masked
    raise ValueError(
        f"identity/time masking did not converge after {MAX_MASK_PASSES} passes"
    )


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    n = 0
    length_totals = {k: 0 for k in ("plain", "short", "long", "masked_short", "masked_long")}
    length_maxima = {k: 0 for k in length_totals}
    with args.input.open(encoding="utf-8") as source, temp.open("w", encoding="utf-8") as out:
        for line in source:
            if not line.strip(): continue
            r = json.loads(line)
            text = str(r.get("text_model", r.get("text", "")) or "")
            title = str(r.get("title_clean_final", r.get("title", "")) or "")
            name = str(r.get("stock_name", "") or str(r.get("stock_id", "")))
            sid = str(r.get("stock_id", ""))
            plain = text
            short = SHORT_TEMPLATE.format(stock_name=name, text=text)
            long = LONG_TEMPLATE.format(stock_name=name, stock_id=sid, title=title, text=text)
            masked_text = mask_identity_and_time(text, name, sid)
            masked_title = mask_identity_and_time(title, name, sid)
            masked_short = MASKED_SHORT_TEMPLATE.format(text=masked_text)
            masked_long = MASKED_LONG_TEMPLATE.format(title=masked_title, text=masked_text)
            fixed_long = LONG_PROMPT + title + "\n公告正文：" + text
            fixed_masked_long = MASKED_LONG_FIXED_TEMPLATE.format(title=masked_title, text=masked_text)
            output_record = {
                "row_index": n,
                "document_id": r.get("document_id"),
                # Alignment metadata is retained outside model-visible text.
                "stock_id_alignment": sid,
                "announcement_date_alignment": r.get("announcement_date"),
                "text_input_5_masked_short": masked_short,
                "text_input_6_masked_long": masked_long,
                "text_input_5_masked_short_sha256": sha(masked_short),
                "text_input_6_masked_long_sha256": sha(masked_long),
                # Preserve short-prompt components for token-level encoding.
                "short_prompt": SHORT_PROMPT,
                "short_title": title,
                "short_body": text,
                "masked_short_prompt": SHORT_PROMPT,
                "masked_short_title": masked_title,
                "masked_short_body": masked_text,
                "long_prompt": LONG_PROMPT,
                "long_title": title,
                "long_body": text,
                "masked_long_prompt": LONG_PROMPT,
                "masked_long_title": masked_title,
                "masked_long_body": masked_text,
                "text_input_7_fixed_long": fixed_long,
                "text_input_8_fixed_masked_long": fixed_masked_long,
                "identity_mask": SUBJECT_MASK,
                "time_mask": TIME_MASK,
                "prompt_version": MASKING_VERSION,
            }
            out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            n += 1
            for key, value in (("plain", plain), ("short", short), ("long", long),
                               ("masked_short", masked_short), ("masked_long", masked_long)):
                length_totals[key] += len(value)
                length_maxima[key] = max(length_maxima[key], len(value))
    os.replace(temp, args.output)
    summary = {"rows": n, "length_policy": "no character-length equalization or truncation at input construction",
               "templates": {"short": SHORT_TEMPLATE, "long": LONG_TEMPLATE,
                             "input_5_masked_short": MASKED_SHORT_TEMPLATE,
                             "input_6_masked_long": MASKED_LONG_TEMPLATE},
               "masking": {"version": MASKING_VERSION,
                           "subject": SUBJECT_MASK, "time": TIME_MASK,
                           "scope": "exact stock name/code, company legal names, and explicit calendar dates/years"},
               "lengths": {k: {"mean": length_totals[k] / n if n else 0.0,
                                "max": length_maxima[k]} for k in length_totals},
               "output": str(args.output)}
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
