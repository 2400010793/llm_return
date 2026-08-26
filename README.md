# 中国市场中的大语言模型新闻收益预测

## 1. 项目目标

复现 Chen、Kelly 和 Xiu 的 *Expected Returns and Large Language Models* 的核心结论，并将研究对象从美国/国际市场扩展到中国股票市场。

核心问题：

1. 中国金融新闻文本是否能够预测未来股票收益？
2. LLM 文本嵌入是否优于传统词袋、词典情绪和 Word2Vec 方法？
3. 新闻信息是否存在价格反应迟滞？
4. 预测结果能否形成具有经济意义的多空投资组合？
5. 中国市场的语言、投资者结构、涨跌停和卖空限制是否会改变原文结论？

> 本项目先做“可复现的最小版本”，再逐步增加模型、数据源和稳健性检验。所有预测必须严格使用当时可获得的信息，避免前视偏差。

## 2. 与第二篇论文的关系

参考论文为 Yang 的 *Investor Sentiment in the Chinese Stock Market*。它不直接复现新闻 LLM 方法，但提供中国市场研究的关键参考：

- 中国股票数据与样本期的组织方式；
- 使用中文投资者评论构造文本情绪指标的方法；
- 使用百度指数而不是 Google Trends 衡量中国投资者关注/情绪；
- 中国市场的行业、地区和投资者结构异质性分析；
- 涨跌停、卖空限制、交易成本和投资者注意力等稳健性问题；
- 时间序列预测、横截面预测、Fama–MacBeth 回归和多空组合评价框架。

本项目将新闻文本作为主信号，将股吧评论情绪、百度搜索指数作为补充信号或控制变量，区分“新闻语气”“投资者情绪”和“投资者关注度”。

## 3. 推荐研究路线

### 阶段 0：研究设计与数据可得性

- 明确研究市场：A 股沪深主板、创业板、科创板是否全部纳入；
- 明确频率：先从日频开始，后续扩展到周频；
- 明确预测期限：次日、未来 5 个交易日、未来 20 个交易日；
- 记录新闻发布时间、来源、语言、关联股票和发布时间时区；
- 确认新闻、行情、股票代码、行业分类和退市股票数据的授权与可得性。

### 阶段 1：构造中国新闻—股票面板

每一行至少包含：

- `article_id`：新闻唯一 ID；
- `stock_id`：关联股票；
- `published_at`：发布时间；
- `headline`、`body`、`language`、`source`；
- `date`：新闻所属交易日；
- `ret_1d`、`ret_5d`、`ret_20d`：未来收益标签；
- 股票价格、成交量、流通市值、行业、停牌和涨跌停状态。

新闻与收益标签的时间对齐是第一优先级：收盘后发布的新闻不能使用当天收盘前收益；非交易日新闻应映射到下一个可交易时点。

### 阶段 2：文本表示与基准模型

先实现可解释、成本较低的基线：

1. 词袋/TF-IDF；
2. 中文金融情绪词典；
3. Word2Vec 或中文句向量；
4. 中文 BERT/RoBERTa；
5. 多语言或中文金融领域预训练模型；
6. API/本地 LLM embedding（在数据合规和成本允许时）。

第一版不叠加复杂神经网络，采用原文类似的“文本表示 + 简单监督预测模型”，以便识别性能究竟来自文本表示还是来自模型复杂度。

### 阶段 3：收益预测模型

对每种文本表示分别估计：

\[
E(r_{i,t+1}\mid x_{i,t}) = x_{i,t}'\theta
\]

建议先使用：

- pooled panel regression；
- Ridge/Lasso/Elastic Net；
- 分期滚动训练或 expanding-window 训练；
- 只使用训练期数据拟合标准化、降维和模型参数。

可加入股票特征作为控制变量，但需要分别报告“仅文本”和“文本 + 股票特征”的结果。

### 阶段 4：投资组合与经济意义

每个交易日或每周：

1. 用历史训练窗口预测股票未来收益；
2. 按预测值分成 5 组或 10 组；
3. 构造高预测组减低预测组的多空组合；
4. 计算平均收益、波动率、Sharpe 比率、最大回撤和换手率；
5. 逐步加入交易成本、涨跌停无法成交、停牌和卖空限制；
6. 与基准信号比较：过去收益、规模、反转、动量、词典情绪和股吧情绪。

中国市场建议至少报告三种组合口径：

- 理论多空组合；
- 仅做多高预测组；
- 考虑卖空限制后的可实施组合。

### 阶段 5：稳健性和机制分析

- 不同预测期限：1、5、20 个交易日；
- 不同新闻新鲜度：新闻提醒、快讯、普通新闻；
- 新闻篇幅、否定词、数字、复杂叙事和重复新闻；
- 大盘、行业和个股固定效应；
- 牛市/熊市、危机期和疫情期；
- 大盘股/小盘股、高/低流动性股票；
- 沪深市场、行业和地区异质性；
- 控制投资者关注度、百度指数、股吧情绪和传统风险因子；
- 安慰剂测试、伪发布时间测试和严格样本外测试；
- 交易成本、涨跌停、停牌、T+1 和卖空约束。

## 4. 当前目录

```text
llm_return/
├── README.md                         # 项目目标、研究计划和复现规范
├── ExpectedReturns_LLMs.pdf          # 第一篇参考论文
├── Phd_thesis_Zhijiao_Yang_20219273_.pdf  # 第二篇参考论文
├── configs/
│   ├── paths.yaml                    # 数据路径和输出路径
│   ├── sample.yaml                   # 样本期、市场和过滤规则
│   └── models.yaml                   # 文本模型和预测模型参数
├── data/
│   ├── raw/                          # 原始数据，只读保存
│   ├── interim/                      # 清洗中间结果
│   └── processed/                    # 可直接建模的数据集
├── references/                       # 论文笔记、变量定义和数据说明
├── notebooks/
│   ├── 01_data_audit.ipynb           # 数据可得性和字段审计
│   ├── 02_news_stock_merge.ipynb     # 新闻—股票匹配与时间对齐
│   ├── 03_text_baselines.ipynb       # 词袋、词典、Word2Vec 基线
│   ├── 04_llm_embeddings.ipynb       # LLM 嵌入生成与质量检查
│   ├── 05_return_prediction.ipynb    # 样本外预测
│   ├── 06_portfolio_backtest.ipynb   # 分组组合和交易成本
│   └── 07_robustness.ipynb           # 稳健性和异质性
├── src/
│   ├── data/
│   │   ├── ingest.py                 # 数据读取
│   │   ├── clean_prices.py            # 行情清洗
│   │   ├── clean_news.py              # 新闻清洗
│   │   └── build_panel.py             # 新闻—股票面板
│   ├── text/
│   │   ├── preprocess_zh.py          # 中文分词、去重和规范化
│   │   ├── sentiment_lexicon.py       # 词典情绪
│   │   ├── embeddings.py              # LLM/句向量接口
│   │   └── quality_checks.py          # 文本质量和泄漏检查
│   ├── models/
│   │   ├── baselines.py               # 传统基线
│   │   ├── return_prediction.py       # 收益预测
│   │   └── walk_forward.py            # 滚动/扩展窗口训练
│   ├── portfolio/
│   │   ├── formation.py               # 分组与持仓形成
│   │   ├── constraints.py             # 涨跌停、停牌、卖空等约束
│   │   └── backtest.py                # 回测
│   └── evaluation/
│       ├── prediction_metrics.py      # MSE、方向准确率等
│       ├── portfolio_metrics.py       # 收益、Sharpe、回撤、换手率
│       └── robustness.py              # 稳健性检验
├── tests/                             # 单元测试和无前视偏差测试
├── reports/                            # 表格、图形和最终报告
└── logs/                               # 运行日志
```

## 5. 建议的第一版最小可行实验

为了避免一开始范围过大，先完成以下实验：

- 市场：沪深 A 股普通股票；
- 频率：日频；
- 文本：中文财经新闻标题 + 正文；
- 基线：TF-IDF + Ridge、中文情绪词典；
- LLM：一种中文 BERT/RoBERTa 句向量模型；
- 预测目标：下一交易日收益；
- 训练方式：至少 3 年训练窗口，滚动样本外预测；
- 组合：五分位多空组合；
- 对照：过去收益、规模、成交量、波动率；
- 输出：预测性能、分组收益、Sharpe、换手率和交易成本敏感性。

只有当最小版本通过时间对齐、前视偏差和安慰剂测试后，才扩展到更多 LLM、长预测期限、百度指数和股吧情绪。

## 6. 关键风险与规范

1. **数据授权：** 新闻正文、百度指数和股吧数据应确认合法授权及使用范围。
2. **前视偏差：** 任何词表、标准化参数、降维结果和模型参数都只能由训练期数据得到。
3. **新闻时间：** 必须区分发布时间和交易日，特别是收盘后、周末和节假日新闻。
4. **股票生存偏差：** 纳入退市股票，避免只保留当前仍上市公司。
5. **交易可行性：** 单独处理涨跌停、停牌、T+1、交易费用和卖空约束。
6. **嵌入可重复性：** 固定模型版本、tokenizer 版本、批处理方式、随机种子和缓存结果。
7. **结果解释：** 将统计显著性与实际可交易性分开报告，不直接把预测收益等同于真实收益。

## 7. 协作方式

后续每次修改建议遵循：

1. 先在 `references/` 中记录变量和假设；
2. 再修改 `configs/`；
3. 先做小样本测试；
4. 通过测试后运行全样本；
5. 将表格和图保存到 `reports/`；
6. 在本 README 中更新已完成事项和待办事项。

## 8. 待办事项

- [ ] 确认新闻数据源、价格数据源和授权范围
- [ ] 确认中国股票市场样本范围和样本期
- [ ] 完成字段字典和数据质量审计
- [ ] 完成新闻与股票的时间对齐规则
- [ ] 实现 TF-IDF 和词典情绪基线
- [ ] 实现第一种中文 LLM 嵌入
- [ ] 完成滚动样本外预测
- [ ] 完成五分位组合回测
- [ ] 加入交易约束和交易成本
- [ ] 加入第二篇论文中的百度指数/股吧情绪参考变量
- [ ] 完成稳健性检验和研究报告

## 9. 下一步执行说明

详细的数据源、模型和安装建议见 [references/data_sources_and_models.md](references/data_sources_and_models.md)。当前不建议立即下载大型模型；应先确认新闻数据授权、字段和发布时间，再用小样本跑通 TF-IDF 基线。

文本表示和 embedding 的分阶段实施方案见 [references/embedding_implementation_plan.md](references/embedding_implementation_plan.md)。原则是先完成 TF-IDF/词典基线，再下载中文 RoBERTa，最后评估 BERT、BGE-M3 和 API embedding。

全面复现路线见 [references/full_replication_plan.md](references/full_replication_plan.md)。当前公开原型使用 GDELT 元数据；正式结果仍应替换为具有授权、历史覆盖和股票实体关联的新闻全文数据。

东方财富、股吧和雪球可作为中国投资者评论情绪的补充数据源，但当前只接收合法 API/网页导出，不绕过登录、验证码、robots.txt 或访问频率限制。相关规范化代码见 `src/data/public_sources.py` 和 `src/data/normalize_public.py`。

## 10. 基础配置使用

项目基础配置已实现：

- 配置文件：[configs/paths.yaml](configs/paths.yaml)、[configs/sample.yaml](configs/sample.yaml)、[configs/models.yaml](configs/models.yaml)；
- 配置加载：[src/config.py](src/config.py)；
- 配置检查命令：[src/cli.py](src/cli.py)。

运行配置检查时使用 `python -m src.cli --prepare-dirs`。该命令只读取 YAML 并创建配置中声明的运行目录，不下载新闻、不下载模型，也不执行回测。

## 11. 新浪财经历史 HTML 采集（本地普通 HTTP）

`scripts/crawl_sina_bfs_fast.py` 现在是低速、单并发、可恢复的普通
`urllib` 采集器，BFS 为默认策略；`scripts/crawl_sina_dfs.py` 使用同一套
解析器和 HTTP 策略提供 DFS 顺序对照。正式采集不使用 Playwright、Selenium、
Chromium、代理轮换或访问控制绕过；robots.txt 无法读取时默认跳过域名。
浏览器脚本仅用于人工发现和验证种子 URL。

### 均匀 frontier 探索

普通 BFS 会优先耗尽最早发现的链接，容易集中在某一个频道或单个历史分支。
采集器现在支持 `--frontier-policy uniform`：每次从当前候选 frontier 中按照
URL 和 `--uniform-seed` 生成的稳定伪随机顺序选择一个页面。它不改变请求频率，
也不绕过任何访问控制；同一 seed 下可复现，并且断点恢复不会因为 Python 随机数
状态变化而改变顺序。该机制是**图结构上的均匀探索**，不是严格的按年份均匀抽样；
若要实现年度或月份配额，应在获得足够历史 URL 后再增加分层采样器。

示例（先用 100 页验证覆盖率，不直接扩大到全量）：

```powershell
$env:PYTHONPATH = "scripts"
python scripts/crawl_sina_bfs_fast.py `
	--strategy bfs `
	--frontier-policy uniform `
	--uniform-seed 20260806 `
	--root https://finance.sina.com.cn/t/34687.html `
	--max-pages 100 `
	--max-depth 2 `
	--pause-seconds 3 `
	--raw-dir data/interim/uniform_probe_raw `
	--articles-output data/interim/uniform_probe_articles.jsonl
```

建议先比较 FIFO 和 uniform 的 `coverage.article_year_counts`、
`coverage.page_quality_counts`、频道路径分布和股票覆盖率，再决定是否进行更大规模
采集。uniform 运行必须固定 `--uniform-seed`；resume 时不能修改该 seed 或 frontier
策略。

### 历史连续扩展与年份种子

500 页结果显示单个种子会被现代页面占据，因此后续不直接扩大全站搜索。采集器支持
`--historical-only-links`，只继续扩展符合 `/t|s|e|y/数字.html` 的旧新浪文章 URL，
从而把历史连续性验证与现代导航探索分开。`--root` 可以重复指定多个历史文章种子，
summary 的 `coverage.root_counts` 会分别记录每个根 URL 的访问数、有效页数、文章数和
404 数量。

推荐先从 5—10 个 2000—2001 年文章种子运行 BFS，`max-depth` 设为 1 或 2；每次使用
独立输出目录，或者在同一批次重复指定 `--root` 并按 `root_counts` 审计。确认历史文章
的年份连续性、股票覆盖率和正文重复率后，再按年份建立独立种子批次。该流程仍不代表
全量覆盖，年份种子只用于可审计的分层扩展。

Windows PowerShell 单页探针：

```powershell
$env:PYTHONPATH = "scripts"
python scripts/crawl_sina_bfs_fast.py `
	--strategy bfs `
	--root https://finance.sina.com.cn/t/34687.html `
	--max-pages 1 `
	--max-depth 0 `
	--max-seconds 60 `
	--pause-seconds 2 `
	--raw-dir data/interim/local_http_probe_raw `
	--articles-output data/interim/local_http_probe_articles.jsonl `
	--manifest-output data/interim/local_http_probe_manifest.jsonl `
	--summary-output data/interim/local_http_probe_summary.json `
	--state-output data/interim/local_http_probe_state.json
```

小规模 BFS：

```powershell
$env:PYTHONPATH = "scripts"
python scripts/crawl_sina_bfs_fast.py `
	--strategy bfs `
	--root https://finance.sina.com.cn/t/34687.html `
	--root https://finance.sina.com.cn/view/general/2000-06-11/36092.html `
	--max-pages 50 --max-depth 2 --max-seconds 600 --pause-seconds 2 `
	--raw-dir data/interim/local_sina_bfs_raw `
	--articles-output data/interim/local_sina_bfs_articles.jsonl `
	--manifest-output data/interim/local_sina_bfs_manifest.jsonl `
	--summary-output data/interim/local_sina_bfs_summary.json `
	--state-output data/interim/local_sina_bfs_state.json --resume
```

DFS 对照只需将入口改为 `scripts/crawl_sina_dfs.py`，并使用同样的输出参数。
默认安全值为 20 页、深度 2、300 秒、请求间隔 2 秒、超时 30 秒、最多重试
1 次、响应上限 10 MB、robots 检查开启；默认仅允许
`finance.sina.com.cn` 和 `cj.sina.com.cn`。`--include-http` 可显式允许旧版
HTTP 链接，`--include-navigation-pages` 可加入低优先级导航页。

采集会追加写入文章 JSONL 和逐页 manifest，并用 state 文件保存队列/栈、已访问
URL、发现顺序、策略、根 URL、白名单、解析器版本和参数。恢复时这些元数据不一致
会明确报错，不会静默混用。原始 HTML 按稳定 URL 哈希保存。真实网络探针不会由
pytest 自动执行。
