# 综合研究报告

本目录保存《中国市场 Prompt Token 与新闻收益预测研究报告》的可复现源文件。

文件职责：

- `report.md`：报告正文；
- `facts.json`：正文和图表使用的冻结事实及来源；
- `build_report.py`：生成图表、PDF 和校验和；
- `figures/`：由冻结事实生成的图；
- `checksums.sha256`：报告源文件、图和 PDF 的 SHA256。

构建命令：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-report.txt
python reports/comprehensive_research_report/build_report.py
```

输出为仓库根目录的 `中国市场PromptToken新闻收益预测研究报告.pdf`。构建过程不访问
网络，也不读取工作区外部数据。修改正文数字时，必须先同步更新 `facts.json` 的值、单位、
证据等级与来源；不能只修改 Markdown 表格。
