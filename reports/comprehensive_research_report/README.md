# 综合研究报告

本目录保存《中国市场 Prompt Token 与新闻收益预测研究报告》的可复现源文件。

报告覆盖新浪与巨潮数据来源、采集和清洗，RoBERTa/BGE-M3/Qwen 模型合同及实测表现，
论文 benchmark，以及 mask/prompt、PCA、硬/软 KMeans、UMAP+HDBSCAN、GMM 和融合
实验。三模型各自四 Prompt token 因子和树模型比较已纳入冻结结果；尚未完成的
token/body 配对与跨模型融合只作为设计列出，不会混入已完成结果。

文件职责：

- `report.md`：报告正文；
- `facts.json`：正文和图表使用的冻结事实及来源；
- `build_report.py`：生成图表、PDF 和校验和；
- `figures/`：由冻结事实生成的图；
- `audits/prompt_cluster_portfolios/`：从底层预测、逐日持仓重算的日均 bp、持仓和成本审计；
- `checksums.sha256`：报告源文件、图和 PDF 的 SHA256。

构建命令：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-report.txt
python reports/comprehensive_research_report/build_report.py
```

需要从原实验目录重新生成聚类持仓审计时运行：

```bash
python scripts/audit_prompt_cluster_portfolios.py \
  --root /mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1
```

输出为仓库根目录的 `中国市场PromptToken新闻收益预测研究报告.pdf`。构建过程不访问
网络，也不读取工作区外部数据。修改正文数字时，必须先同步更新 `facts.json` 的值、单位、
证据等级与来源；不能只修改 Markdown 表格。
