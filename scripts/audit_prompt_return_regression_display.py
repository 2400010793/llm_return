#!/usr/bin/env python3
"""Freeze the two existing Prompt-to-return result tables used by the report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/comprehensive_research_report/audits/prompt_return_regressions"


def main() -> None:
    historical = json.loads(
        (ROOT / "reports/prompt_cluster_token_comparison.json").read_text(encoding="utf-8")
    )
    four = pd.DataFrame(historical["linear_and_hard"])
    four = four[
        four["method_family"].eq("linear_ridge")
        & four["variant"].eq("masked_short")
    ][["prompt_zh", "model", "rankic"]].copy()
    four["model"] = four["model"].map({"roberta": "RoBERTa", "bge_m3": "BGE-M3"})
    four = four.rename(columns={"prompt_zh": "prompt", "rankic": "rankic_next_day"})
    four = four.sort_values(["prompt", "model"], kind="stable")

    axes = pd.read_csv(ROOT / "reports/aligned_factor_clusters/all_results.csv")
    axes = axes[
        axes["representation"].eq("token") & axes["method"].eq("pca_ridge")
    ][["axis", "target", "test_year", "rankic", "top20_ls", "n_test"]].copy()
    axes["top20_ls_bp"] = axes["top20_ls"] * 10_000
    axes = axes.sort_values("rankic", ascending=False, kind="stable")

    payload = {
        "standalone_neutral_prompt_status": (
            "Embedding/similarity complete or partial by model; no completed common-return "
            "regression for the six standalone neutral prompts."
        ),
        "four_direction_protocol": (
            "Sina legacy common panel; masked_short target span; next_day_return; "
            "2018-2026 strict 6+2+1; no-cluster linear Ridge."
        ),
        "semantic_axis_protocol": (
            "18 aligned high/neutral/low prompts collapsed to six direction axes; RoBERTa "
            "masked_short target span; forward_compounded_return_3d; 2018-2023 train, "
            "2024-2025 validation, 2026 test; PCA+Ridge."
        ),
        "four_direction_rows": int(len(four)),
        "semantic_axis_rows": int(len(axes)),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    four.to_csv(OUT / "four_direction_masked_next_day.csv", index=False)
    axes.to_csv(OUT / "semantic_axes_return3d_2026.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
