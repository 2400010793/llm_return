# 研究报告索引

面向研究主管和后续接手者的完整报告为仓库根目录的
`中国市场PromptToken新闻收益预测研究报告.pdf`。可复现源文件、冻结事实和构建脚本位于
`comprehensive_research_report/`。

报告按研究问题组织，不按任务编号组织。历史专题结果仍保留在 Lustre 工作区；主报告只
引用已经完成、能够追溯且口径明确的结果。未完成任务、单年发现和平台报错结果会明确标注，
不会与正式跨年证据混为一谈。

```bash
python reports/comprehensive_research_report/build_report.py
```

构建过程只读取仓库内的 `report.md` 和 `facts.json`，不读取未来收益、外部网络、模型 API
或未发布的大型实验目录。
