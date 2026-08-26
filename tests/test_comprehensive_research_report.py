from __future__ import annotations

import importlib.util
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
    assert "尚无同一收益标签上的公平 RankIC 横表" in source
