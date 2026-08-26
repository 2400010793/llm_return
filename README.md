# 中国市场中的大语言模型新闻收益预测

> 当前研究快照：2026-08-26。项目已经从通用复现原型推进到新浪/巨潮
> 双数据源、RoBERTa/BGE-M3/Qwen 三模型、prompt 目标 token 与正文表示的
> 严格样本外比较。最新事实状态、已完成结果、未完成任务和接手命令见
> [完整交接文档](docs/HANDOFF_2026-08-26.md)；论文依据见
> [论文到实验的映射](references/paper_to_experiment_map.md)；日常操作见
> [运行手册](docs/RUNBOOK.md)。

## 当前主线

当前预注册比较只使用四个 `masked_short` prompt：`盈利`、`收益`、
`超额收益`、`亏损`，并在完全共同新闻上比较三个模型的目标 span token
表示与正文表示。所有监督模型只使用训练期拟合的 PCA32，不做原始维回归；
基础参数固定为 Ridge alpha 100、KMeans k 6、soft shrinkage 500、temperature
0.5。新浪采用 6 年训练、2 年验证、1 年测试；巨潮基础模型采用 3 年历史、
1 年测试，并保留历史窗口内部的 1 年 OOS 预测供二级融合训练。

截至快照时的可核查数据：

| 项目 | 状态 |
|---|---:|
| 新浪全量清洗面板 | 796,553 条 |
| 巨潮全量面板 | 903,665 条 |
| 中性六 prompt 新浪 RoBERTa | 256/256 shards |
| 中性六 prompt 巨潮 RoBERTa | 255/256 shards，缺 shard-225 |
| 中性六 prompt 新浪 BGE-M3 | 189/256 shards，任务已主动取消 |
| 中性六 prompt 巨潮 BGE-M3 | 78/256 shards，任务已主动取消 |
| 三模型四 prompt PCA32 公平比较 | 已提交，等待 Fairshare 调度；尚无结果 |

使用下面的只读命令可以重新生成当前数据与 embedding 状态：

```bash
python scripts/audit_research_handoff.py --include-slurm
```

本仓库只保存源码、配置、测试和小型研究说明。授权数据、模型权重、PDF、
embedding、任务快照和运行报告保留在工作区/Lustre，不进入 Git；它们通过
清单、SHA256 和绝对路径在交接文档中追溯。

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
├── README.md                          # 项目目标、研究计划和复现规范
├── pyproject.toml                     # Python 包与测试配置
├── configs/
│   ├── paths.yaml                     # 数据路径和输出路径
│   ├── sample.yaml                    # 样本期、市场和过滤规则
│   └── models.yaml                    # 文本模型和预测模型参数
├── data/
│   ├── raw/                           # 原始数据，只读保存
│   ├── interim/                       # 清洗和采集中间结果
│   └── processed/                     # 可直接建模的数据与冻结嵌入
├── references/                        # 论文笔记、变量定义和实施计划
├── src/
│   ├── data/                           # 清洗、去重、标签、面板和嵌入读取
│   ├── text/                           # 文本预处理、表示和嵌入接口
│   ├── models/                         # 可复用的模型与无泄漏预处理
│   │   ├── return_prediction.py        # 股票—日聚合、评价、Ridge 选参
│   │   ├── dimension_reduction.py      # 训练窗拟合的 PCA/SVD
│   │   └── paper_pipeline.py           # 小型 TF-IDF 论文基线
│   ├── portfolio/
│   │   ├── formation.py                # 每日等权分位数组合
│   │   ├── strategy.py                 # EWCT、手续费、持仓和交易约束
│   │   └── performance.py              # 毛/净收益、Sharpe、换手和回撤
│   └── evaluation/
│       ├── prediction_metrics.py       # MSE、OOS R²、方向准确率、Rank IC
│       ├── classification.py           # 分类指标与稳定性统计
│       └── artifacts.py                # 内容寻址、锁、原子写和恢复
├── scripts/                            # 薄 CLI、manifest、审计和 Slurm 入口
├── tests/                              # 单元测试和无前视偏差测试
├── task_records/                       # 长任务提交与审计记录
├── reports/                            # 表格、预测、组合和最终报告
└── logs/                               # 运行日志
```

### 代码职责

- `src/` 只放可导入、可测试、与命令行无关的研究逻辑；多个实验需要的代码应优先放在这里。
- `scripts/` 只负责参数解析、数据路径、滚动窗口编排和任务提交，不重复实现指标、组合或预处理。
- `src/models/return_prediction.py` 是连续收益任务的统一入口；公告级模型输出先聚合为股票—日，再进行验证选参和测试评价。
- `src/portfolio/` 独立于模型，既可用于 TF-IDF，也可用于冻结 embedding 或其他预测器。
- `src/evaluation/artifacts.py` 统一管理实验身份、文件指纹、单写者锁、原子保存和断点恢复。

### 完整策略回测

`scripts/run_portfolio_strategy.py` 接收股票—日预测文件，输出逐日收益、逐股持仓、逐笔交易和汇总指标。论文模式实现次日开盘建仓、顶部/底部五分位、多空组合、10/20 bps 大/小盘股成本，以及 EWCT 权重递推。A 股模式支持买卖双边佣金、卖出印花税、滑点、停牌/涨跌停可交易标记、融券标记和仅做多组合。

```bash
python scripts/run_portfolio_strategy.py \
  reports/example.stock_day_predictions.parquet \
  --output-dir reports/strategy/example \
  --cost-model paper --gammas 1.0
```

`gamma < 1` 会延长持仓，因此必须通过 `--market-data` 提供稠密的股票—交易日收益；程序默认在持仓收益缺失时终止，不会静默填零。论文 10/20 bps 拆分还需要 `--market-cap-column` 或 `--small-stock-column`，否则仅采用论文的大盘股 10 bps 简化回退口径，并在汇总文件中记录该限制。

`scripts/collect_strategy_ohlc.py` 可按预测文件中的股票池下载并缓存前复权 OHLC，构造 O2O、C2C、VWAP 代理收益，以及停牌、IPO 初期和开盘涨跌停交易标记。当前 RoBERTa 最终测试样本的 815 只股票已全部接入；修正交易时点和成本后的结果见 [2026 可执行交易回测审计](reports/strategy/real_trading_audit_2026.md) 与 [完整指标表](reports/strategy/roberta_corrected_v2_2026_execution_comparison.csv)。这些结果是历史可执行性回测，不是实盘成交记录。

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
- [x] 完成五分位组合与持仓级净收益回测
- [x] 实现 EWCT、交易成本及可配置交易约束，并接入测试股票池的稠密 OHLC/交易状态
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
