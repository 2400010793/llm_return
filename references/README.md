# 参考文献与研究记录

- `papers_manifest.tsv`：五份工作区论文的文件名、书目信息、页数和 SHA256；PDF 不入 Git。
- `paper_to_experiment_map.md`：论文原方法、指标与本项目 benchmark/扩展的逐项映射。
- `data_sources_and_models.md`：新浪、巨潮、东方财富、雪球、行情和三模型的当前合同。
- `embedding_implementation_plan.md`：pooled/token embedding 产物及执行历史。
- `dynamic_prompt_replication_protocol_v1.md`：prompt、mask、span 和滚动协议。
- `full_replication_plan.md`：从论文基线到中国市场交易约束的完整复现路线。

研究口径发生变化时，先更新数据/模型/prompt 合同，再修改 runner；结论数字必须同步进入
冻结 facts 或可追溯结果表，不能只追加到叙述性 Markdown。
