from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "comprehensive_research_report"


def load_builder():
    path = REPORT_DIR / "build_report.py"
    spec = importlib.util.spec_from_file_location("comprehensive_report_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_facts_match_report_contract() -> None:
    builder = load_builder()
    facts = builder.load_facts()
    assert facts["datasets"]["sina_full"]["rows"] == 796_553
    assert facts["datasets"]["cninfo_full"]["rows"] == 903_665
    assert facts["datasets"]["sina_legacy_aligned"]["rows"] == 75_894
    for dataset in facts["datasets"].values():
        if "year_counts" in dataset:
            assert sum(dataset["year_counts"].values()) == dataset["rows"]
    assert facts["direction_prompt_results"]["folds_complete"] == 144
    cninfo_collection = facts["collection"]["cninfo"]
    assert cninfo_collection["candidate_universe_rows"] == 1_000
    assert cninfo_collection["completed_universe_rows"] == 816
    assert sum(cninfo_collection["completed_by_exchange"].values()) == 816
    assert sum(cninfo_collection["completed_code_prefixes"].values()) == 816
    assert "not all A shares" in cninfo_collection["technical_scope"]
    assert cninfo_collection["selection_regimes"]["2010-2017"].startswith("all announcements")
    assert cninfo_collection["selection_regimes"]["2018-2026"].startswith("focus announcements")
    assert cninfo_collection["clean_panel_rows_2010_2017_all_announcements"] == 553_088
    assert cninfo_collection["clean_panel_rows_2018_2026_focus_announcements"] == 350_577
    assert (
        cninfo_collection["clean_panel_rows_2010_2017_all_announcements"]
        + cninfo_collection["clean_panel_rows_2018_2026_focus_announcements"]
        == facts["datasets"]["cninfo_full"]["rows"]
    )
    cleaning = facts["data_cleaning"]
    assert cleaning["sina"]["accepted_source_articles"] == 167_713
    assert cleaning["sina"]["expanded_stock_article_rows"] == 796_553
    assert cleaning["cninfo"]["generic_combined_clean_rows"] == 903_816
    assert cleaning["cninfo"]["frozen_research_panel_rows"] == 903_665
    assert "151-row difference" in cleaning["cninfo"]["version_warning"]
    alignment = facts["date_alignment"]
    assert "strictly after" in alignment["entry_date"]
    assert "does not use a publication-time close cutoff" in alignment["main_contract"]
    runtime = facts["embedding_runtime"]
    arrays = {(row["dataset"], row["model"]): row for row in runtime["arrays"]}
    assert arrays[("Sina", "RoBERTa")]["completed_shards"] == 256
    assert arrays[("Sina", "RoBERTa")]["observed_array_wall_seconds"] == 16_097
    assert arrays[("CNINFO", "RoBERTa")]["completed_shards"] == 255
    assert arrays[("Sina", "BGE-M3")]["cancelled_shards"] == 67
    assert arrays[("CNINFO", "BGE-M3")]["cancelled_shards"] == 178
    assert runtime["qwen_reference"]["elapsed_seconds"] == 18_715
    assert abs(runtime["qwen_reference"]["rows_per_second"] - 3.085) < 1e-12
    original = facts["direction_prompt_results"]["masked_short_linear_ridge"]
    assert len(original) == 8
    assert {(row["model"], row["prompt"]) for row in original} == {
        (model, prompt)
        for model in ("RoBERTa", "BGE-M3")
        for prompt in ("盈利", "收益", "超额收益", "亏损")
    }
    mask_pairs = facts["direction_prompt_results"]["mask_linear_rankic_pairs"]
    assert len(mask_pairs) == 8
    assert sum(row["prompt_mean_delta"] > 0 for row in mask_pairs) == 8
    assert sum(row["target_span_delta"] > 0 for row in mask_pairs) == 7
    assert abs(
        sum(row["prompt_mean_delta"] for row in mask_pairs) / len(mask_pairs) - 0.0032365857
    ) < 1e-10
    assert abs(
        sum(row["target_span_delta"] for row in mask_pairs) / len(mask_pairs) - 0.0029729179
    ) < 1e-10
    assert next(
        row for row in mask_pairs if row["prompt"] == "超额收益" and row["model"] == "RoBERTa"
    )["target_span_delta"] < 0
    new_axes = facts["clustering"]["new_axes_single_2026"]
    comparison = new_axes["horizon_vs_volatility_token_rankic"]
    assert len(comparison) == 4
    assert all(row["horizon"] > row["volatility"] for row in comparison)
    assert new_axes["horizon_wins_methods"] == 4
    assert (
        facts["token_body"]["models"][2]["token_rankic"]
        < facts["token_body"]["models"][2]["whole_text_rankic"]
    )
    representations = facts["token_body"]["historical_four_prompt_masked_pca128"]
    roberta = next(row for row in representations if row["model"] == "RoBERTa")
    bge = next(row for row in representations if row["model"] == "BGE-M3")
    assert roberta["prompt_minus_span"] > 0
    assert bge["prompt_minus_span"] < 0
    qwen = facts["token_body"]["qwen_masked_pca32"]
    assert qwen["article_mean_rankic"] > qwen["prompt_mean_rankic"] > qwen["return_token_rankic"]
    layouts = facts["embedding_layouts"]
    assert layouts["roberta_bge_m3"]["model_visible_separator"].startswith("none")
    assert "double newline" in layouts["qwen3_embedding_8b"]["model_visible_separator"]
    hard = facts["clustering"]["hard_kmeans_vs_ridge"]
    assert hard["hard_net_daily_bp"] < hard["linear_net_daily_bp"]
    assert hard["net_daily_bp_wins"] == 7
    assert hard["average_holdings"] < 3
    assert {row["prompt"] for row in hard["by_prompt"]} == {"盈利", "收益", "超额收益", "亏损"}
    soft = facts["direction_prompt_results"]["soft_long_only_prompt_leaders"]
    assert len(soft) == 4
    assert all(row["average_nominal_holdings"] > 80 for row in soft)
    assert all(13 < row["average_effective_holdings"] < 16 for row in soft)
    assert all(abs(row["average_current_top20_names"] - 1.10) < 0.01 for row in soft)
    gamma = facts["direction_prompt_results"]["soft_gamma_sensitivity"]
    assert len(gamma) == 10
    assert gamma[0]["net_daily_bp"] > 0
    assert gamma[-1]["net_daily_bp"] < 0
    assert gamma[0]["average_current_top20_names"] == gamma[-1]["average_current_top20_names"]
    density = facts["clustering"]["bge_loss_umap_hdbscan"]
    assert density["umap_hdbscan_long_short_bp"] > density["pca_ridge_long_short_bp"]
    assert density["costs_included"] is False
    roadmap = facts["research_roadmap"]
    assert roadmap["prompt_factor"]["current_evidence"]["roberta_token_cross_prompt_spearman"] == 0.653
    assert roadmap["prompt_factor"]["current_evidence"]["bge_m3_token_cross_prompt_spearman"] == 0.892
    assert "paired 6+2+1 RankIC delta positive in at least 6/9 test years" in roadmap["prompt_factor"]["acceptance"]
    assert "Top20 overlap and late-fusion incremental RankIC" in roadmap["model_correlation"]["next_measurements"]
    assert "CSI 300" in roadmap["regime_expansion"]["macro_news"]
    fair = facts["fair_three_model_four_prompt_token_tree"]
    assert fair["no_raw_embedding_regression"] is True
    assert fair["no_cross_model_fusion"] is True
    assert fair["sina"]["valid_rankic_days"] == 67
    assert fair["cninfo"]["valid_rankic_days"] == 625
    assert fair["sina"]["cross_prompt_spearman"]["BGE-M3"] == 0.8771
    assert fair["cninfo"]["cross_prompt_spearman"]["Qwen3-Embedding-8B"] == 0.6573
    assert (
        fair["cninfo"]["cross_prompt_spearman"]["BGE-M3"]
        > fair["cninfo"]["cross_prompt_spearman"]["RoBERTa"]
        > fair["cninfo"]["cross_prompt_spearman"]["Qwen3-Embedding-8B"]
    )
    cninfo_qwen = fair["key_results"]["cninfo"]["Qwen3-Embedding-8B"]
    assert abs(cninfo_qwen["tree_rankic"] - 0.0527539464) < 1e-10
    assert cninfo_qwen["tree_long_short_bp"] < cninfo_qwen["best_single_long_short_bp"]
    sina_bge = fair["key_results"]["sina"]["BGE-M3"]
    assert sina_bge["equal_weight_rankic"] > sina_bge["tree_rankic"]
    return_comparison = facts["prompt_return_comparison"]
    assert "no completed common-return regression" in return_comparison["standalone_neutral_status"]
    axes = {row["axis"]: row for row in return_comparison["semantic_axes_pca_ridge_token"]}
    assert set(axes) == {"确定性", "期限收益", "波动率", "流动性", "估值", "冲击"}
    assert abs(axes["波动率"]["rankic"] - 0.0429753126) < 1e-10
    assert abs(axes["波动率"]["top20_ls_bp"] - 14.4825706) < 1e-7
    stock_mask = facts["stock_token_mask_results"]
    assert stock_mask["representation"] == "stock_span"
    assert len(stock_mask["rows"]) == 8
    assert stock_mask["aggregate"]["RoBERTa"]["positive_prompt_pairs"] == "3/4"
    assert stock_mask["aggregate"]["BGE-M3"]["positive_prompt_pairs"] == "1/4"
    assert abs(stock_mask["aggregate"]["all"]["delta"] - 0.0007011713) < 1e-10
    body = facts["body_mean_baseline"]
    assert body["prompt_conditioned"] is True
    assert len(body["rows"]) == 6
    assert body["pooling"]["RoBERTa"] == "body_mean"
    assert body["pooling"]["Qwen3-Embedding-8B"] == "article_mean"
    assert abs(next(row for row in body["rows"] if row["dataset"] == "新浪" and row["model"] == "RoBERTa")["rankic"] - 0.0594172720) < 1e-10
    token_body = facts["fair_token_body_comparison"]
    assert len(token_body["rows"]) == 6
    sina_qwen = next(row for row in token_body["rows"] if row["dataset"] == "新浪" and row["model"].startswith("Qwen"))
    assert abs(sina_qwen["rankic_delta"] + 0.0156430648) < 1e-10
    assert len(token_body["sina_prompt_rows"]) == 8


def test_report_has_aligned_sections_and_no_unresolved_placeholders() -> None:
    source = (REPORT_DIR / "report.md").read_text(encoding="utf-8")
    for heading in range(1, 13):
        assert f"## {heading}." in source
    assert "## 附录 F：指标和方法速查" in source
    assert "{{" not in source
    assert "796,553" in source
    assert "903,665" in source
    assert "0.05360" in source
    assert "0.05680" in source
    assert "两模型四方向 Prompt 的同口径收益统计" in source
    assert "新 Prompt 在同一收益标签上的公平比较" in source
    assert "平均只持有 2.80 只股票" in source
    assert "3.10--3.63 净 bp/交易日" in source
    assert "当日 Q5 平均" in source
    assert "权重有效持仓只有 13.9--15.1 只" in source
    assert "Gamma 敏感性" in source
    assert "成本后为 -0.55 bp" in source
    assert "Prompt 的已确认作用与因果边界" in source
    assert "Prompt mean short" in source
    assert "8/8 改善" in source
    assert "+0.00324/+0.00226" in source
    assert "-0.00207" in source
    assert "精确“股票”Token 的 Mask 前后结果" in source
    assert "0.057157" in source
    assert "0.040961" in source
    assert "+0.000701" in source
    assert "stock_token_mask_rankic_deltas.png" in source
    assert "加入 Prompt 相对无 Prompt 的因果增量" in source
    assert "Prompt 与正文如何隔断" in source
    assert "没有额外 separator token" in source
    assert "不能命名为 `body_mean`" in source
    assert "0.05731" in source
    assert "0.03630" in source
    assert "0.05547" in source
    assert "正文 `body_mean` 基线（所有 Prompt 结果之前）" in source
    assert "0.059417" in source
    assert "0.075959" in source
    assert "Prompt-conditioned body baseline" in source
    assert "Prompt Token 相对正文基线的直接结果" in source
    assert "0.060316" in source
    assert "-0.015643" in source
    assert "Prompt token RankIC" in source
    assert "尚无同一收益标签上的公平 RankIC 横表" in source
    assert "主线一：证明不同 Prompt 真的产生不同因子" in source
    assert "三模型相关性应该如何理解" in source
    assert "主线二：Regime、全股票与宏观新闻扩展" in source
    assert "全股票横截面" in source
    assert "CSI 300" in source
    assert "同长度、同位置但与标签无关的中文短语" in source
    assert "巨潮目前爬取范围：技术范围与已完成范围不是一回事" in source
    assert "沪市 64、深市 752" in source
    assert "新浪为什么必须在本地，以及实际爬取逻辑" in source
    assert "本地浏览器发现 + 本地普通 HTTP 扩展 + 本地去重/清洗" in source
    assert "这里的 `N × P × D` 是数组形状" in source
    assert "`N × 6 × 768`" in source
    assert "六条单独中性 Prompt" in source
    assert "+14.48" in source
    assert "不能用 0.05623 和 0.05486 直接判断" in source
    assert "2010--2017 抓取这 816 只股票的**全部公告**" in source
    assert "2018--2026 使用这 816 只股票的" in source
    assert "**重点公告**" in source
    assert "2018 年筛选断点" in source
    assert "`full` 文件名解释为全时期全公告" in source
    assert "从原始文本到可交易日期：清洗与对齐合同" in source
    assert "167,713 篇源文章" in source
    assert "903,816 中间档不等于 903,665 冻结面板" in source
    assert "entry_date        = 严格晚于公告自然日的首个交易所交易日" in source
    assert "Embedding 用时、并行资源和估算边界" in source
    assert "4时28分17秒，完整" in source
    assert "不是全量完成时间" in source
    assert "单次 pooled pass" in source
    assert "71.7 小时" in source
    assert "三模型四 Prompt Token 的公平比较" in source
    assert "0.052754" in source
    assert "0.8771" in source
    assert "树模型没有稳定创造增量" in source
    assert "新浪仅有 67 个有效 RankIC 日" in source


def test_pdf_builder_resolves_the_simplified_chinese_font_by_family() -> None:
    source = (REPORT_DIR / "build_report.py").read_text(encoding="utf-8")
    assert 'font-family: "Noto Sans CJK SC"' in source
    assert 'html { font-family: "Noto Sans CJK SC", sans-serif;' in source
    assert "ReportSans" not in source
    assert "NotoSansCJK-Regular.ttc" not in source
    assert "references/embedding_runtime_audit_20260826.md" in source


def test_prompt_mask_rankic_audit_matches_frozen_facts() -> None:
    audit_dir = REPORT_DIR / "audits" / "prompt_mask"
    audit = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    facts = load_builder().load_facts()
    frozen = facts["direction_prompt_results"]["mask_linear_rankic_summary"]
    assert len(audit["rows"]) == 16
    assert audit["summary"]["prompt_mean"]["wins"] == frozen["prompt_mean"]["wins"] == 8
    assert audit["summary"]["target_span"]["wins"] == frozen["target_span"]["wins"] == 7
    assert abs(
        audit["summary"]["prompt_mean"]["mean_delta"]
        - frozen["prompt_mean"]["mean_delta"]
    ) < 1e-10
    assert abs(
        audit["summary"]["target_span"]["mean_delta"]
        - frozen["target_span"]["mean_delta"]
    ) < 1e-10
    assert (audit_dir / "rankic_pairs.csv").stat().st_size > 0


def test_prompt_representation_audit_matches_frozen_facts() -> None:
    audit_dir = REPORT_DIR / "audits" / "prompt_representations"
    audit = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    facts = load_builder().load_facts()["token_body"]
    frozen_models = {row["model"]: row for row in facts["historical_four_prompt_masked_pca128"]}
    for row in audit["four_prompt_model_averages"]:
        frozen = frozen_models[row["model"]]
        for key in ("prompt_mean_rankic", "direction_span_rankic", "prompt_minus_span"):
            assert abs(row[key] - frozen[key]) < 1e-10
    qwen = {row["representation"]: row for row in audit["qwen"]}
    assert abs(qwen["article_mean"]["rankic"] - facts["qwen_masked_pca32"]["article_mean_rankic"]) < 1e-10
    assert abs(qwen["prompt_mean"]["rankic"] - facts["qwen_masked_pca32"]["prompt_mean_rankic"]) < 1e-10
    assert abs(qwen["return_token"]["rankic"] - facts["qwen_masked_pca32"]["return_token_rankic"]) < 1e-10
    assert "full_mean" in audit["true_body_mean_status"]
    for name in ("four_prompt_masked_rankic.csv", "qwen_masked_pca32_rankic.csv"):
        assert (audit_dir / name).stat().st_size > 0


def test_prompt_return_display_audit_matches_frozen_facts() -> None:
    audit_dir = REPORT_DIR / "audits" / "prompt_return_regressions"
    summary = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["four_direction_rows"] == 8
    assert summary["semantic_axis_rows"] == 6
    assert "no completed common-return regression" in summary["standalone_neutral_prompt_status"]
    with (audit_dir / "semantic_axes_return3d_2026.csv").open(encoding="utf-8", newline="") as handle:
        axes = {row["axis"]: row for row in csv.DictReader(handle)}
    assert abs(float(axes["volatility"]["rankic"]) - 0.0429753126) < 1e-10
    assert abs(float(axes["volatility"]["top20_ls_bp"]) - 14.4825706) < 1e-7
    with (audit_dir / "four_direction_masked_next_day.csv").open(encoding="utf-8", newline="") as handle:
        four = list(csv.DictReader(handle))
    assert len(four) == 8


def test_per_model_prompt_token_tree_audit_matches_frozen_facts() -> None:
    audit_path = REPORT_DIR / "audits" / "per_model_prompt_token_tree" / "summary.csv"
    with audit_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    lookup = {(row["dataset"], row["model"], row["method"]): row for row in rows}
    facts = load_builder().load_facts()["fair_three_model_four_prompt_token_tree"]
    qwen = lookup[("cninfo", "qwen3_embedding_8b", "tree_selected")]
    assert abs(
        float(qwen["rank_ic"])
        - facts["key_results"]["cninfo"]["Qwen3-Embedding-8B"]["tree_rankic"]
    ) < 1e-10
    bge = lookup[("sina", "bge_m3", "equal_weight")]
    assert abs(
        float(bge["rank_ic"])
        - facts["key_results"]["sina"]["BGE-M3"]["equal_weight_rankic"]
    ) < 1e-10


def test_portfolio_audit_is_frozen_and_matches_report_facts() -> None:
    audit_dir = REPORT_DIR / "audits" / "prompt_cluster_portfolios"
    summary = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    facts = load_builder().load_facts()
    hard = facts["clustering"]["hard_kmeans_vs_ridge"]
    assert summary["protocols_are_not_level_comparable"] is True
    assert summary["linear_hard"]["pairs"] == 16
    assert abs(
        summary["linear_hard"]["hard_minus_linear_net_daily_bp_mean"]
        - hard["hard_minus_linear_net_daily_bp"]
    ) < 1e-5
    assert len(summary["soft_long_only"]["leaders_by_prompt"]) == 4
    for name in (
        "linear_hard_daily.csv",
        "soft_long_only_daily.csv",
        "soft_gamma_sensitivity.csv",
        "umap_hdbscan_daily.csv",
    ):
        assert (audit_dir / name).stat().st_size > 0


def test_stock_token_mask_audit_matches_frozen_facts() -> None:
    audit_path = REPORT_DIR / "audits" / "stock_token_mask" / "summary.csv"
    with audit_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    facts = load_builder().load_facts()["stock_token_mask_results"]
    assert len(rows) == len(facts["rows"]) == 8
    assert {(row["model"], row["prompt"]) for row in rows} == {
        (row["model"], row["prompt"]) for row in facts["rows"]
    }
    assert sum(float(row["masked_minus_short_rankic"]) > 0 for row in rows) == 4
