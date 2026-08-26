"""Conservative cleaning helpers for archived Sina Finance article text."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone as datetime_timezone
import hashlib
import re
from typing import Any

from src.text.preprocess_zh import clean_text


_LEADING_PROMOS = (
    re.compile(
        r"^【线索征集令！】.*?爆料联系邮箱：\s*[^ ]+\s*",
        re.DOTALL,
    ),
    re.compile(r"^新浪财经Level2：A股极速看盘\s+新浪财经App：直播上线\s+博主一对一指导\s*"),
    re.compile(r"^新浪财经App：直播上线\s+博主一对一指导(?:\s+新浪港股APP：实时行情\s+独家内参)?\s*"),
    re.compile(r"^热点栏目\s+自选股\s+数据中心\s+行情中心\s+资金流向\s+模拟交易\s+客户端\s*"),
    re.compile(
        r"^.{0,500}?热点栏目\s+自选股\s+数据中心\s+行情中心\s+"
        r"资金流向\s+模拟交易\s+客户端\s*"
    ),
    re.compile(r"^热点栏目\s+资金流向\s+千股千评\s+个股诊断\s+最新评级\s+模拟交易\s+客户端\s*"),
    re.compile(r"^登录新浪财经APP\s+搜索【信披】查看更多考评等级\s*"),
    re.compile(
        r"^新浪财经\s+语音播报\s+缩小字体\s+放大字体\s+收藏\s+微博\s+微信\s+分享\s+腾讯QQ\s+QQ空间\s*"
    ),
    re.compile(
        r"^新浪网\s+作者\s+.{1,60}?\s+(?:优质财经领域创作者\s+)?"
        r"缩小字体\s+放大字体\s+收藏\s+微博\s+微信\s+分享\s+(?:\d+\s+)?腾讯QQ\s+QQ空间\s*"
    ),
    re.compile(
        r"^.{0,500}?炒股就看\s+金麒麟分析师研报\s*，权威，专业，及时，全面，"
        r"助您挖掘潜力主题机会！\s*"
    ),
    re.compile(
        r"^.{0,300}?语音播报\s+缩小字体\s+放大字体(?:\s+收藏)?\s+"
        r"微博\s+微信\s+分享(?:\s+\d+)?\s+腾讯QQ\s+QQ空间\s*"
    ),
    re.compile(
        r"^7x24\s*小时全球实时财经新闻\s+直播\s+坚持做最好的财经直播报道，"
        r"给百姓最真的财经动态。\s+\d{2}月\d{2}日\s+\d{2}:\d{2}\s*",
        re.IGNORECASE,
    ),
)
_FOOTER_MARKERS = (
    "THE_END",
    "进入 【新浪财经股吧】",
    "进入【新浪财经股吧】",
    "海量资讯、精准解读，尽在新浪财经APP",
    "新浪财经意见反馈留言板",
    "新浪科技意见反馈留言板",
    "新浪意见反馈留言板",
    "股市直播 图文直播间 视频直播间",
    "相关新闻 加载中 点击加载更多",
    "新浪简介 ┊",
    "新浪简介 |",
    "特别声明：以上文章内容仅代表作者本人观点",
    "分享到:",
    "分享到：",
    "@@title@@",
    "查看更多董秘问答>>",
    "文章关键词：",
    "我要反馈 新浪直播 百位牛人在线解读股市热点",
    "炒股开户享福利",
    "股市跌了别害怕！",
    "反弹行情下的专属投资礼包！",
    "股市回暖，抄底炒股先开户！",
)
_TAIL_EDITOR = re.compile(
    r"\s+责任编辑[：:]\s*[^ ]{0,30}"
    r"(?=\s+(?:SF\d+|文章关键词|我要反馈|相关专题|APP专享直播|热门推荐)|$)"
)
_TRAILING_DISCLAIMERS = (
    "免责声明：自媒体综合提供的内容均源自自媒体",
    "新浪声明：新浪网登载此文出于传递更多信息之目的",
    "新浪声明：此消息系转载自新浪合作媒体",
)
_TRAILING_EDITOR_BLOCK = re.compile(
    r"\s+(?:[（(【]?责任编辑[：:]\s*[^ ]{1,30}[）)】]?"
    r"(?:\s+主编[：:]\s*[^ ]{1,30})?"
    r"|编辑\s*[|｜：:]\s*[^ ]{1,30}(?:\s+校对[：:]\s*[^ ]{1,30})?"
    r"(?:\s+审核[：:]\s*[^ ]{1,30})?)\s*$"
)
_DATE_PREFIX = re.compile(
    r"^(?:20\d{2}年\d{1,2}月\d{1,2}日|20\d{2}[-/]\d{1,2}[-/]\d{1,2}日?)"
    r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\s*"
)
_DATE_ANYWHERE = re.compile(
    r"(?:20\d{2}年\d{1,2}月\d{1,2}日|20\d{2}[-/]\d{1,2}[-/]\d{1,2}日?)"
    r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"
)
_SHARE_CHROME = re.compile(
    r"^.{0,600}?新浪财经APP\s+缩小字体\s+放大字体\s+收藏\s+"
    r".{0,100}?腾讯QQ\s+QQ空间\s*"
)

TARGET_PROMPT_TEMPLATE = (
    "任务：请仅基于下面的新浪财经新闻，判断该新闻对目标股票{stock_name}"
    "（股票代码：{stock_id}）未来短期收益方向的影响。"
    "新闻可能同时涉及多只股票；请只评估目标股票，不得把其他公司的信息错误归因于目标股票。"
    "只允许使用新闻发布时已经公开的信息，不得使用发布后的股价、收益或其他未来信息。"
)


def _trim_modern_header(title: str, body: str) -> tuple[str, list[str], int]:
    """Remove a leading Sina navigation/header block when it is unambiguous."""
    original = body
    actions: list[str] = []
    search_limit = min(len(body), 800)
    header = body[:search_limit]
    if "> 正文" in header:
        positions = [match.end() for match in re.finditer(re.escape(title), header)] if title else []
        if positions:
            body = body[positions[-1] :].lstrip()
            actions.append("modern_header")
        else:
            displayed_date = _DATE_ANYWHERE.search(header)
            if displayed_date:
                body = body[displayed_date.start() :].lstrip()
                actions.append("modern_header")

    date_match = _DATE_PREFIX.match(body)
    if date_match:
        body = body[date_match.end() :].lstrip()
        actions.append("published_display")

    share_match = _SHARE_CHROME.match(body)
    if share_match:
        body = body[share_match.end() :].lstrip()
        actions.append("share_chrome")

    changed = True
    while changed:
        changed = False
        for pattern in _LEADING_PROMOS:
            match = pattern.match(body)
            if match:
                body = body[match.end() :].lstrip()
                actions.append("leading_promo")
                changed = True
    return body, actions, len(original) - len(body)


def _trim_footer(body: str) -> tuple[str, list[str], int]:
    """Cut only recognizable Sina page furniture near the end of an article."""
    original_length = len(body)
    actions: list[str] = []
    candidates: list[tuple[int, str]] = []
    for marker in _FOOTER_MARKERS:
        position = body.find(marker)
        if position >= 0:
            candidates.append((position, marker))

    for marker in _TRAILING_DISCLAIMERS:
        position = body.find(marker)
        if position >= 0:
            candidates.append((position, marker))

    # Match an editor attribution only when followed by known Sina furniture.
    for match in _TAIL_EDITOR.finditer(body):
        if match.start() >= 10:
            candidates.append((match.start(), "责任编辑"))
            break

    if not candidates:
        return body, actions, 0
    position, marker = min(candidates)
    body = body[:position].rstrip()
    editor_match = _TRAILING_EDITOR_BLOCK.search(body)
    if editor_match:
        body = body[: editor_match.start()].rstrip()
        actions.append("trailing_editor")
    actions.append(f"footer:{marker}")
    return body, actions, original_length - len(body)


def clean_sina_article_text(title: Any, body: Any) -> dict[str, Any]:
    """Normalize one article while preserving its substantive wording."""
    title_clean = clean_text(title)
    body_normalized = clean_text(body)
    body_clean, header_actions, header_chars = _trim_modern_header(title_clean, body_normalized)
    body_clean, footer_actions, footer_chars = _trim_footer(body_clean)
    body_clean = clean_text(body_clean)
    text = f"标题：{title_clean}。正文：{body_clean}" if title_clean and body_clean else title_clean or body_clean
    return {
        "title_clean": title_clean,
        "body_clean": body_clean,
        "text": text,
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "body_raw_chars": len(str(body or "")),
        "body_clean_chars": len(body_clean),
        "header_trimmed_chars": header_chars,
        "footer_trimmed_chars": footer_chars,
        "cleaning_actions": header_actions + footer_actions,
    }


def build_sina_target_text(
    *,
    stock_id: str,
    stock_name: str,
    title: str,
    body: str,
) -> dict[str, str]:
    """Build model-visible text that identifies one target in a multi-stock story."""
    display_name = stock_name.strip() or "目标公司"
    target_prompt = TARGET_PROMPT_TEMPLATE.format(
        stock_name=display_name,
        stock_id=stock_id,
    )
    text_model = f"{target_prompt}\n新闻标题：{title}\n新闻正文：{body}"
    return {
        "target_prompt": target_prompt,
        "text_model": text_model,
        "text_model_hash": hashlib.sha256(text_model.encode("utf-8")).hexdigest(),
    }


def normalize_sina_timestamp(value: Any, timezone: str = "+08:00") -> str:
    """Return a second-resolution ISO timestamp, assuming China time if naive."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("empty timestamp")

    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        match = re.fullmatch(
            r"(20\d{2})(?:年|[-/])(\d{1,2})(?:月|[-/])(\d{1,2})(?:日)?"
            r"(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?",
            raw,
        )
        if not match:
            raise
        year, month, day, hour, minute, second = match.groups()
        parsed = datetime(
            int(year),
            int(month),
            int(day),
            int(hour or 0),
            int(minute or 0),
            int(second or 0),
        )
    if parsed.tzinfo is None:
        offset = re.fullmatch(r"([+-])(\d{2}):(\d{2})", timezone)
        if not offset:
            raise ValueError(f"invalid timezone offset: {timezone}")
        sign, hours, minutes = offset.groups()
        delta = timedelta(hours=int(hours), minutes=int(minutes))
        if sign == "-":
            delta = -delta
        parsed = parsed.replace(tzinfo=datetime_timezone(delta))
    return parsed.isoformat(timespec="seconds")
