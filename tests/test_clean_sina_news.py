import json
from pathlib import Path
import subprocess
import sys

from src.data.clean_sina_news import (
    build_sina_target_text,
    clean_sina_article_text,
    normalize_sina_timestamp,
)


def test_clean_sina_modern_page_chrome() -> None:
    title = "公司净利润同比增长20%"
    body = (
        "证券 > 正文 行情 股吧 新闻 外汇 新三板 "
        f"{title} {title} 2024年03月22日 08:30 中证网 "
        "新浪财经APP 缩小字体 放大字体 收藏 微博 微信 分享 腾讯QQ QQ空间 "
        "公司实现营业收入100亿元，净利润同比增长20%，经营现金流改善。 "
        "海量资讯、精准解读，尽在新浪财经APP 责任编辑：张三"
    )

    result = clean_sina_article_text(title, body)

    assert result["body_clean"] == "公司实现营业收入100亿元，净利润同比增长20%，经营现金流改善。"
    assert "modern_header" in result["cleaning_actions"]
    assert "share_chrome" in result["cleaning_actions"]
    assert result["footer_trimmed_chars"] > 0


def test_clean_sina_modern_share_variant_and_disclaimer() -> None:
    body = (
        "上海证券报 新浪财经APP 缩小字体 放大字体 收藏 微博 分享 MD 微信 腾讯QQ QQ空间 "
        "公司完成重大资产重组，主营业务保持稳定。 "
        "新浪声明：新浪网登载此文出于传递更多信息之目的，并不意味着赞同其观点。"
    )

    result = clean_sina_article_text("重大资产重组", body)

    assert result["body_clean"] == "公司完成重大资产重组，主营业务保持稳定。"
    assert result["cleaning_actions"] == [
        "share_chrome",
        "footer:新浪声明：新浪网登载此文出于传递更多信息之目的",
    ]


def test_clean_sina_legacy_footer_without_changing_financial_language() -> None:
    body = (
        "公司预计净利润同比下降82%，不存在重大遗漏，提请投资者注意投资风险。 "
        "THE_END 进入 【新浪财经股吧】 讨论 我要反馈 保存网页"
    )

    result = clean_sina_article_text("业绩预减公告", body)

    assert result["body_clean"] == "公司预计净利润同比下降82%，不存在重大遗漏，提请投资者注意投资风险。"
    assert "投资风险" in result["text"]
    assert "新浪财经股吧" not in result["text"]


def test_clean_sina_short_article_before_large_legacy_footer() -> None:
    body = (
        "据报道，公司与另一家公司正在讨论合并，最终方案尚未确定。 "
        "进入 【新浪财经股吧】 讨论 责任编辑：编辑 相关阅读 聚焦 应用中心 新浪公益"
    )

    result = clean_sina_article_text("公司讨论合并", body)

    assert result["body_clean"] == "据报道，公司与另一家公司正在讨论合并，最终方案尚未确定。"


def test_clean_sina_7x24_toolbar() -> None:
    body = (
        "新浪财经 语音播报 缩小字体 放大字体 收藏 微博 微信 分享 腾讯QQ QQ空间 "
        "公司全年净利润61.4亿元人民币。 责任编辑：编辑 我要反馈"
    )

    result = clean_sina_article_text("全年业绩", body)

    assert result["body_clean"] == "公司全年净利润61.4亿元人民币。"


def test_clean_sina_7x24_live_page_chrome() -> None:
    body = (
        "7x24 小时全球实时财经新闻 直播 坚持做最好的财经直播报道，给百姓最真的财经动态。 "
        "07月13日 13:42 公司直线涨停，成交明显放量。 "
        "分享到: 微博 微信 QQ空间 下一条快讯将在 ?? 秒后 到达新浪财经APP "
        "最先掌握财经7x24快讯 就在新浪财经APP 立即前往 扫码下载 链接财富"
    )

    result = clean_sina_article_text("公司直线涨停_7x24小时财经新闻_新浪网", body)

    assert result["body_clean"] == "公司直线涨停，成交明显放量。"
    assert "leading_promo" in result["cleaning_actions"]
    assert "footer:分享到:" in result["cleaning_actions"]


def test_clean_sina_legacy_template_and_promotion_tail() -> None:
    body = (
        "公司前三季度营业收入同比增长，归母净利润同比下降。 "
        "@@title@@ @@teacher_name@@：@@title@@ 热门推荐"
    )
    result = clean_sina_article_text("前三季度业绩", body)
    assert result["body_clean"] == "公司前三季度营业收入同比增长，归母净利润同比下降。"

    promoted = clean_sina_article_text(
        "业绩快讯",
        "公司预计本年度净利润增长20%。 炒股开户享福利，入金抽188元红包，100%中奖！",
    )
    assert promoted["body_clean"] == "公司预计本年度净利润增长20%。"


def test_clean_sina_jin_qilin_header_and_qa_footer() -> None:
    body = (
        "热点栏目 自选股 数据中心 炒股就看 金麒麟分析师研报 ，权威，专业，及时，全面，"
        "助您挖掘潜力主题机会！ 公司回应称在手订单保持稳定。 "
        "查看更多董秘问答>> 免责声明：本信息由新浪财经从公开信息中摘录。"
    )

    result = clean_sina_article_text("公司回应订单情况", body)

    assert result["body_clean"] == "公司回应称在手订单保持稳定。"


def test_clean_sina_legacy_market_header_and_keyword_footer() -> None:
    body = (
        "热点栏目 资金流向 千股千评 个股诊断 最新评级 模拟交易 客户端 "
        "公司中标水处理项目，预计对未来业绩产生积极影响。 "
        "(责任编辑：finet) 文章关键词： 公司 中标 水处理 我要反馈 "
        "新浪直播 百位牛人在线解读股市热点，带你挖掘板块龙头 收起"
    )

    result = clean_sina_article_text("公司中标水处理项目", body)

    assert result["body_clean"] == "公司中标水处理项目，预计对未来业绩产生积极影响。"
    assert "leading_promo" in result["cleaning_actions"]
    assert "trailing_editor" in result["cleaning_actions"]


def test_clean_sina_promotion_before_market_header_and_feedback_footer() -> None:
    body = (
        "新浪财经评选活动正在进行，欢迎参与投票。【点击投票】 "
        "热点栏目 自选股 数据中心 行情中心 资金流向 模拟交易 客户端 "
        "公司发布年度业绩，净利润同比增长。 "
        "我要反馈 新浪直播 百位牛人在线解读股市热点，带你挖掘板块龙头 收起"
    )

    result = clean_sina_article_text("公司发布年度业绩", body)

    assert result["body_clean"] == "公司发布年度业绩，净利润同比增长。"


def test_clean_sina_disclosure_app_header() -> None:
    body = (
        "登录新浪财经APP 搜索【信披】查看更多考评等级 "
        "公司实施股份回购，回购方案符合相关规定。"
    )

    result = clean_sina_article_text("股份回购进展", body)

    assert result["body_clean"] == "公司实施股份回购，回购方案符合相关规定。"


def test_clean_sina_empty_self_media_shell() -> None:
    body = (
        "新浪网 作者 某财经作者 缩小字体 放大字体 收藏 微博 微信 分享 腾讯QQ QQ空间 "
        "特别声明：以上文章内容仅代表作者本人观点，不代表新浪网观点或立场。 阅读排行榜 相关新闻"
    )

    result = clean_sina_article_text("文章标题", body)

    assert result["body_clean"] == ""


def test_clean_sina_keeps_plain_article_unchanged() -> None:
    body = "公司发布新产品。市场有风险，投资需谨慎，但该句属于报道原文。"

    result = clean_sina_article_text("新产品发布", body)

    assert result["body_clean"] == body
    assert result["cleaning_actions"] == []


def test_normalize_sina_timestamp_assumes_china_timezone() -> None:
    assert normalize_sina_timestamp("2016-04-01 06:54") == "2016-04-01T06:54:00+08:00"
    assert normalize_sina_timestamp("2026-08-07T14:26:30+08:00") == "2026-08-07T14:26:30+08:00"


def test_normalize_sina_timestamp_accepts_archived_chinese_and_slash_dates() -> None:
    assert normalize_sina_timestamp("2019年8月20") == "2019-08-20T00:00:00+08:00"
    assert normalize_sina_timestamp("2014年04月25日 10:16") == "2014-04-25T10:16:00+08:00"
    assert normalize_sina_timestamp("2026/4/20") == "2026-04-20T00:00:00+08:00"


def test_build_sina_target_text_focuses_one_stock() -> None:
    result = build_sina_target_text(
        stock_id="688235",
        stock_name="百济神州",
        title="两家创新药企业发布业绩",
        body="百济神州和荣昌生物均披露了最新经营情况。",
    )

    assert "目标股票百济神州（股票代码：688235）" in result["target_prompt"]
    assert "不得把其他公司的信息错误归因于目标股票" in result["target_prompt"]
    assert result["text_model"].endswith("新闻正文：百济神州和荣昌生物均披露了最新经营情况。")
    assert len(result["text_model_hash"]) == 64


def test_clean_sina_cli_expands_multi_stock_article(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "article_id": "article-1",
                "screen_stock_ids": ["688235", "688331"],
                "title": "两家创新药企业发布业绩",
                "body": "百济神州和荣昌生物均披露了最新经营情况，公司经营数据出现明显改善。",
                "published_at": "2026-08-07T07:03:41+08:00",
                "coverage_year": 2026,
                "source": "sina_finance",
                "content_type": "financial_news",
                "url": "https://finance.sina.com.cn/example.shtml",
                "body_truncated": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    universe = tmp_path / "stocks.csv"
    universe.write_text(
        "stock_id,stock_name,as_of_date\n688235,百济神州,2026-08-05\n688331,荣昌生物,2026-08-05\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "cleaned"

    subprocess.run(
        [
            sys.executable,
            "scripts/clean_sina_content_quality.py",
            str(source),
            "--output-dir",
            str(output_dir),
            "--stock-universe",
            str(universe),
            "--min-body-chars",
            "20",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "sina_stock_target_clean.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["stock_id"] for row in rows] == ["688235", "688331"]
    assert len({row["document_id"] for row in rows}) == 2
    assert all(row["is_multi_stock_article"] for row in rows)
    assert all(row["target_stock_count"] == 2 for row in rows)
    assert "百济神州（股票代码：688235）" in rows[0]["target_prompt"]
    assert "荣昌生物（股票代码：688331）" in rows[1]["target_prompt"]
