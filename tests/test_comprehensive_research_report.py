from __future__ import annotations

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
    original = facts["direction_prompt_results"]["masked_short_linear_ridge"]
    assert len(original) == 8
    assert {(row["model"], row["prompt"]) for row in original} == {
        (model, prompt)
        for model in ("RoBERTa", "BGE-M3")
        for prompt in ("盈利", "收益", "超额收益", "亏损")
    }
    new_axes = facts["clustering"]["new_axes_single_2026"]
    comparison = new_axes["horizon_vs_volatility_token_rankic"]
    assert len(comparison) == 4
    assert all(row["horizon"] > row["volatility"] for row in comparison)
    assert new_axes["horizon_wins_methods"] == 4
    assert facts["token_body"]["models"][2]["token_rankic"] < facts["token_body"]["models"][2]["body_rankic"]
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
    assert "加入 Prompt 的证据边界" in source
    assert "尚无同一收益标签上的公平 RankIC 横表" in source


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
