"""Insert completed CNINFO tree-stacking results into the main report."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


START = "<!-- CNINFO_TREE_STACK_RESULTS_START -->"
END = "<!-- CNINFO_TREE_STACK_RESULTS_END -->"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    metrics = pd.read_csv(args.result_dir / "test_2026_metrics.csv")
    manifest = json.loads((args.result_dir / "manifest.json").read_text(encoding="utf-8"))
    importance = pd.read_csv(args.result_dir / "feature_importance.csv")

    indexed = metrics.set_index("method")
    tree = indexed.loc["tree_selected"]
    comparisons = []
    for baseline in ("best_single", "rank_equal", "ridge"):
        row = indexed.loc[baseline]
        comparisons.append({
            "baseline": baseline,
            "tree_minus_baseline_rank_ic": tree["rank_ic_mean"] - row["rank_ic_mean"],
            "tree_minus_baseline_top20_bp": tree["top20_mean_bp"] - row["top20_mean_bp"],
            "tree_minus_baseline_long_short_bp": (
                tree["long_short_mean_bp"] - row["long_short_mean_bp"]
            ),
        })
    comparison = pd.DataFrame(comparisons)
    top_importance = (
        importance[importance["method"].eq("tree_selected")]
        .nlargest(10, "importance")
        .loc[:, ["fitted_family", "feature", "importance"]]
    )
    beats = all(comparison["tree_minus_baseline_rank_ic"] > 0)
    conclusion = (
        "验证期选出的树模型在2026年RankIC上超过全部三个基线，树聚合形成了封存样本外增量。"
        if beats else
        "验证期选出的树模型未在2026年RankIC上同时超过最佳单因子、等权和Ridge，因此不能认定树聚合有效。"
    )
    block = "\n".join([
        START,
        "### 树模型封存测试结果",
        "",
        f"验证期选择的树模型为 `{manifest['selected_specs']['tree_selected']['selected_family']}`；以下全部为2026年一次性封存测试结果。",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "相对基线增量：",
        "",
        comparison.to_markdown(index=False, floatfmt=".6f"),
        "",
        "验证期所选树模型的主要一级因子：",
        "",
        top_importance.to_markdown(index=False, floatfmt=".6f"),
        "",
        conclusion,
        "",
        "完整验证搜索、预测、因子相关和特征重要性位于 `prompt_positive_v1/results/tree_stacking_cninfo_oos_v1/`。",
        END,
    ])
    current = args.report.read_text(encoding="utf-8")
    if START in current and END in current:
        before, remainder = current.split(START, 1)
        _, after = remainder.split(END, 1)
        updated = before.rstrip() + "\n\n" + block + after
    else:
        updated = current.rstrip() + "\n\n" + block + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.report.parent, delete=False
    ) as handle:
        handle.write(updated)
        temporary = Path(handle.name)
    os.replace(temporary, args.report)
    print(json.dumps({
        "report": str(args.report), "tree_family": tree["fitted_family"],
        "tree_rank_ic": tree["rank_ic_mean"], "beats_all_rankic_baselines": beats,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
