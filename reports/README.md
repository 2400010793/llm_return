# 研究报告索引

面向研究主管和后续接手者的完整报告为仓库根目录的
`中国市场PromptToken新闻收益预测研究报告.pdf`。可复现源文件、冻结事实和构建脚本位于
`comprehensive_research_report/`。

报告按“论文与 benchmark、数据来源和采集、模型与表示、mask/prompt、监督结果、
PCA/聚类、融合、风险和下一步”组织，不按 Slurm 任务编号组织。历史专题结果保留在
Lustre；主报告只引用能回指 CSV/manifest 的结果。未完成任务、单年发现和平台报错
会显式标注，不与跨年证据混为一谈。

主要入口：

- `comprehensive_research_report/report.md`：综合报告正文；
- `comprehensive_research_report/facts.json`：冻结事实和来源；
- `aligned_factor_clusters/`：语义轴四种聚类方法的配对结果；
- `aligned_extended_regression/`：估值、波动率和流动性标签结果；
- `qwen_prompt_embedding_notebook_masked_short/`：Qwen PCA/KMeans/GMM/HDBSCAN；
- `strategy/` 和 `token_embedding_sharpe_20260821/`：组合与成本。

```bash
python reports/comprehensive_research_report/build_report.py
```

构建过程只读取仓库内的 `report.md` 和 `facts.json`，不读取未来收益、外部网络、模型 API
或未发布的大型实验目录。
