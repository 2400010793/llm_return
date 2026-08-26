#!/usr/bin/env python3
"""Build the comprehensive prompt-token research report from frozen facts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "report.md"
FACTS = HERE / "facts.json"
FIGURES = HERE / "figures"
AUDITS = HERE / "audits"
OUTPUT = ROOT / "中国市场PromptToken新闻收益预测研究报告.pdf"
CHECKSUMS = HERE / "checksums.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_facts() -> dict:
    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    if facts.get("schema_version") != 1:
        raise ValueError("unsupported facts schema")
    for key in ("sina_full", "cninfo_full", "sina_legacy_aligned"):
        item = facts["datasets"][key]
        if "unique_row_index" in item and item["rows"] != item["unique_row_index"]:
            raise ValueError(f"non-unique row_index in {key}")
        if sum(item["year_counts"].values()) != item["rows"]:
            raise ValueError(f"year counts do not sum to rows in {key}")
    prompts = facts["direction_prompt_results"]
    if prompts["folds_complete"] != prompts["folds_expected"]:
        raise ValueError("direction prompt folds are incomplete")
    return facts


def configure_plotting() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Matplotlib resolves the first face of this TTC as the JP family;
            # WeasyPrint uses the explicit SC family below for report text.
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
        }
    )


def generate_figures(facts: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    configure_plotting()
    FIGURES.mkdir(parents=True, exist_ok=True)

    leaders = facts["direction_prompt_results"]["rankic_leaders"]
    labels = [f'{row["prompt"]}\n{row["model"]}' for row in leaders]
    values = [row["rankic"] for row in leaders]
    colors = ["#2a9d8f", "#2a9d8f", "#1f4e79", "#1f4e79", "#1f4e79"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=180)
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_ylabel("日均 RankIC")
    ax.set_title("四方向 Prompt 软聚类的代表性样本外结果")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(values) * 1.22)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.0007, f"{value:.4f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "direction_prompt_rankic.png", bbox_inches="tight")
    plt.close(fig)

    gamma_rows = facts["direction_prompt_results"]["soft_gamma_sensitivity"]
    gamma = [row["gamma"] for row in gamma_rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), dpi=180)
    axes[0].plot(gamma, [row["gross_daily_bp"] for row in gamma_rows], marker="o", label="毛收益")
    axes[0].plot(gamma, [row["cost_daily_bp"] for row in gamma_rows], marker="o", label="成本")
    axes[0].plot(gamma, [row["net_daily_bp"] for row in gamma_rows], marker="o", label="净收益")
    axes[0].axhline(0, color="#687580", linewidth=0.7)
    axes[0].set_xlabel("EWCT gamma")
    axes[0].set_ylabel("bp / 交易日")
    axes[0].set_title("收益与成本")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(
        gamma,
        [row["average_nominal_holdings"] for row in gamma_rows],
        marker="o",
        label="名义持仓",
    )
    axes[1].plot(
        gamma,
        [row["average_effective_holdings"] for row in gamma_rows],
        marker="o",
        label="有效持仓",
    )
    axes[1].plot(
        gamma,
        [row["average_current_top20_names"] for row in gamma_rows],
        linestyle="--",
        label="当日 Q5",
    )
    axes[1].set_xlabel("EWCT gamma")
    axes[1].set_ylabel("股票数")
    axes[1].set_title("持仓衰减")
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    fig.suptitle("四 Prompt 固定 leader 的执行层 gamma 敏感性", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "soft_gamma_sensitivity.png", bbox_inches="tight")
    plt.close(fig)

    mask_rows = facts["direction_prompt_results"]["mask_linear_rankic_pairs"]
    labels = [
        f'{row["prompt"]}\n{row["model"].replace("RoBERTa", "RoB").replace("BGE-M3", "BGE")}'
        for row in mask_rows
    ]
    prompt_delta = [row["prompt_mean_delta"] for row in mask_rows]
    span_delta = [row["target_span_delta"] for row in mask_rows]
    x = np.arange(len(mask_rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 3.45), dpi=180)
    ax.bar(x - width / 2, prompt_delta, width, label="Prompt mean", color="#1f4e79")
    ax.bar(x + width / 2, span_delta, width, label="目标 span", color="#c45a3c")
    ax.axhline(0, color="#687580", linewidth=0.7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("masked-short - short RankIC")
    ax.set_title("同模型、同表示的 Mask 配对增量")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "mask_rankic_deltas.png", bbox_inches="tight")
    plt.close(fig)

    rows = facts["token_body"]["models"]
    names = [row["model"].replace("Qwen3-Embedding-8B", "Qwen3") for row in rows]
    body = [row["whole_text_rankic"] for row in rows]
    token = [row["token_rankic"] for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 3.3), dpi=180)
    ax.bar(x - width / 2, body, width, label="全文/Article 表示", color="#1f4e79")
    ax.bar(x + width / 2, token, width, label="目标 token", color="#c45a3c")
    ax.set_xticks(x, names)
    ax.set_ylabel("日均 RankIC")
    ax.set_title("Token 与正文的现有配对结果")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "token_body_rankic.png", bbox_inches="tight")
    plt.close(fig)

    completion = facts["neutral_embedding_completion"]
    labels = [f'{row["dataset"]} {row["model"]}' for row in completion]
    ratios = [row["complete"] / row["expected"] for row in completion]
    fig, ax = plt.subplots(figsize=(7.0, 3.1), dpi=180)
    bars = ax.barh(labels, ratios, color=["#2a9d8f", "#1f4e79", "#9a7b4f", "#9a7b4f"])
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("完成分片比例")
    ax.set_title("六个中性 masked-short Prompt 的 Embedding 完整性")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, row in zip(bars, completion):
        ax.text(
            min(bar.get_width() + 0.015, 1.01),
            bar.get_y() + bar.get_height() / 2,
            f'{row["complete"]}/{row["expected"]}',
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(FIGURES / "neutral_embedding_completion.png", bbox_inches="tight")
    plt.close(fig)

    comparison = facts["clustering"]["new_axes_single_2026"]["horizon_vs_volatility_token_rankic"]
    methods = [
        row["method"].replace("hard KMeans + Ridge", "hard KMeans")
        .replace("UMAP8 + HDBSCAN", "UMAP + HDBSCAN")
        for row in comparison
    ]
    horizon = [row["horizon"] for row in comparison]
    volatility = [row["volatility"] for row in comparison]
    x = np.arange(len(comparison))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=180)
    ax.bar(x - width / 2, horizon, width, label="期限收益轴", color="#1f4e79")
    ax.bar(x + width / 2, volatility, width, label="波动率轴", color="#c45a3c")
    ax.axhline(0, color="#687580", linewidth=0.7)
    ax.set_xticks(x, methods)
    ax.set_ylabel("2026 测试 RankIC")
    ax.set_title("同新闻、同三日收益标签下的新 Prompt 方向轴")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "new_prompt_return_axis_rankic.png", bbox_inches="tight")
    plt.close(fig)

    hard_rows = facts["clustering"]["hard_kmeans_vs_ridge"]["by_prompt"]
    labels = [row["prompt"] for row in hard_rows]
    linear = [row["linear_net_daily_bp"] for row in hard_rows]
    hard = [row["hard_net_daily_bp"] for row in hard_rows]
    x = np.arange(len(hard_rows))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 3.3), dpi=180)
    ax.bar(x - width / 2, linear, width, label="PCA + Ridge", color="#1f4e79")
    ax.bar(x + width / 2, hard, width, label="硬 KMeans + Ridge", color="#c45a3c")
    ax.set_xticks(x, labels)
    ax.set_ylabel("净 bp / 信号日")
    ax.set_title("四 Prompt 的集中 Top20% 多头执行结果")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "hard_kmeans_daily_bp.png", bbox_inches="tight")
    plt.close(fig)


def render(facts: dict) -> None:
    import markdown
    from weasyprint import CSS, HTML

    generate_figures(facts)
    source = SOURCE.read_text(encoding="utf-8")
    source = source.replace("<!-- PAGEBREAK -->", '<div class="page-break"></div>')
    body = markdown.markdown(
        source,
        extensions=["tables", "sane_lists", "toc", "footnotes", "md_in_html"],
    )
    document = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        "<title>中国市场 Prompt Token 与新闻收益预测研究报告</title>"
        f"</head><body>{body}</body></html>"
    )
    css = CSS(
        string=r"""
@page {
  size: A4;
  margin: 15mm 15mm 18mm 15mm;
  @bottom-left {
    content: "中国市场 Prompt Token 与新闻收益预测研究";
    color: #687580;
    font-family: "Noto Sans CJK SC";
    font-size: 7pt;
    border-top: 0.35pt solid #cbd5dc;
    padding-top: 2.4mm;
  }
  @bottom-right {
    content: counter(page);
    color: #687580;
    font-family: "Noto Sans CJK SC";
    font-size: 7pt;
    border-top: 0.35pt solid #cbd5dc;
    padding-top: 2.4mm;
  }
}
@page:first {
  @bottom-left { content: none; border: none; }
  @bottom-right { content: none; border: none; }
}
* { box-sizing: border-box; }
html { font-family: "Noto Sans CJK SC", sans-serif; color: #20272d; }
body { margin: 0; font-size: 8.4pt; line-height: 1.58; letter-spacing: 0; }
.cover {
  min-height: 250mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
  border-top: 4mm solid #173c5e;
  border-bottom: 0.6mm solid #173c5e;
  padding: 20mm 10mm 28mm 10mm;
  break-after: page;
}
.cover h1 { font-size: 28pt; line-height: 1.25; margin: 0 0 8mm 0; color: #173c5e; }
.cover .subtitle { font-size: 13pt; color: #3f657f; margin-bottom: 20mm; }
.cover .meta { margin-top: auto; color: #66737c; font-size: 9pt; }
h1 { color: #173c5e; font-size: 24pt; line-height: 1.25; letter-spacing: 0; }
h2 { margin: 5mm 0 3mm 0; color: #173c5e; font-size: 15.5pt; line-height: 1.35; page-break-after: avoid; }
h3 { margin: 3.5mm 0 2mm 0; color: #315f7d; font-size: 11.3pt; line-height: 1.4; page-break-after: avoid; }
h4 { margin: 2.7mm 0 1.5mm 0; color: #3d596d; font-size: 9.5pt; page-break-after: avoid; }
p { margin: 0 0 2.6mm 0; text-align: left; orphans: 3; widows: 3; }
strong { color: #18364d; font-weight: 700; }
ul, ol { margin: 1mm 0 3mm 0; padding-left: 5.6mm; }
li { margin: 0 0 1.3mm 0; padding-left: 0.6mm; }
blockquote { margin: 2.5mm 0 3.2mm 0; padding: 2.8mm 3.4mm; border-left: 1.8pt solid #315f7d; background: #edf3f6; color: #18364d; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 0.88em; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; margin: 1.5mm 0 3.4mm 0; font-size: 6.55pt; line-height: 1.35; page-break-inside: auto; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th { padding: 1.45mm 1.2mm; color: white; background: #173c5e; font-weight: 600; text-align: left; vertical-align: top; }
td { padding: 1.3mm 1.2mm; border-bottom: 0.35pt solid #cad4db; vertical-align: top; overflow-wrap: anywhere; }
tbody tr:nth-child(even) td { background: #f5f7f8; }
tbody tr:last-child td { border-bottom: 0.7pt solid #315f7d; }
.toc { margin: 3mm 0 4mm 0; }
.toc ul { list-style: none; padding-left: 0; }
.toc ul ul { padding-left: 5mm; }
.toc a { color: #315f7d; text-decoration: none; }
.page-break { break-before: page; page-break-before: always; }
.figure { margin: 3.5mm auto 4mm auto; text-align: center; page-break-inside: avoid; }
.figure img { max-width: 96%; max-height: 112mm; }
.figure p { color: #5a6670; font-size: 7.4pt; text-align: center; }
.note { background: #f4f7f8; border: 0.4pt solid #cad4db; padding: 2.5mm 3mm; margin: 2mm 0 3mm 0; }
.references { font-size: 6.2pt; line-height: 1.2; }
.references ol { margin: 1mm 0 0 0; }
.references li { margin-bottom: 0.6mm; }
"""
    )
    HTML(string=document, base_url=str(HERE)).write_pdf(
        str(OUTPUT), stylesheets=[css], pdf_identifier=True
    )


def write_checksums() -> None:
    files = [
        SOURCE,
        FACTS,
        Path(__file__),
        ROOT / "scripts/audit_prompt_mask_rankic.py",
        ROOT / "scripts/audit_prompt_representation_rankic.py",
        ROOT / "scripts/audit_prompt_return_regression_display.py",
        ROOT / "references/embedding_runtime_audit_20260826.md",
        OUTPUT,
    ]
    files.extend(sorted(FIGURES.glob("*.png")))
    files.extend(sorted(path for path in AUDITS.rglob("*") if path.is_file()))
    CHECKSUMS.write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(ROOT)}" for path in files) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    facts = load_facts()
    render(facts)
    write_checksums()
    print(OUTPUT)
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
