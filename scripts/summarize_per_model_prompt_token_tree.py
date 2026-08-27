#!/usr/bin/env python3
"""Summarize independent per-model prompt-token tree results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = ("roberta", "bge_m3", "qwen3_embedding_8b")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rows = []
    audits = []
    for dataset in ("sina", "cninfo"):
        for model in MODELS:
            root = args.tree_root / dataset / model
            if not (root / "COMPLETED").is_file():
                raise FileNotFoundError(root / "COMPLETED")
            metrics = pd.read_csv(root / "metrics.csv")
            rows.append(metrics)
            audits.append(json.loads((root / "manifest.json").read_text(encoding="utf-8")))
    summary = pd.concat(rows, ignore_index=True)
    output = args.tree_root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "per_model_prompt_tree_metrics.csv", index=False)
    aggregate = summary.groupby(["dataset", "model", "method"], as_index=False).agg(
        rank_ic=("rank_ic", "mean"), rank_ic_ir=("rank_ic_ir", "mean"),
        positive_years=("rank_ic", lambda x: int((x > 0).sum())), years=("test_year", "nunique"),
        top20_mean_bp=("top20_mean_bp", "mean"), bottom20_mean_bp=("bottom20_mean_bp", "mean"),
        long_short_mean_bp=("long_short_mean_bp", "mean"),
    )
    aggregate.to_csv(output / "per_model_prompt_tree_summary.csv", index=False)
    manifest = {"format_version": "per_model_prompt_token_tree_summary_v1", "datasets": ["sina", "cninfo"], "models": list(MODELS), "inputs": audits}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.report:
        block = "<!-- PER_MODEL_PROMPT_TOKEN_TREE_V1_START -->\n## 独立模型内四 Prompt Token 树模型\n\n"
        block += "本节每个模型单独使用盈利、收益、超额收益、亏损四个 token 因子；不包含其他模型因子，不做三模型融合。一级因子统一为 PCA32-Ridge 样本外预测，二级分别评估 XGBoost、LightGBM、CatBoost，参数和滚动窗口固定。\n\n"
        block += "详表见 `per_model_prompt_tree_metrics.csv` 和 `per_model_prompt_tree_summary.csv`。\n<!-- PER_MODEL_PROMPT_TOKEN_TREE_V1_END -->"
        current = args.report.read_text(encoding="utf-8") if args.report.exists() else ""
        start, end = "<!-- PER_MODEL_PROMPT_TOKEN_TREE_V1_START -->", "<!-- PER_MODEL_PROMPT_TOKEN_TREE_V1_END -->"
        if start in current and end in current:
            prefix, rest = current.split(start, 1); _, suffix = rest.split(end, 1)
            current = prefix.rstrip() + "\n\n" + block + suffix
        else:
            current = current.rstrip() + "\n\n" + block + "\n"
        args.report.write_text(current, encoding="utf-8")
    (output / "COMPLETED").write_text("per_model_prompt_token_tree_summary_v1\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
