# 论文依据与实验映射

## 原始材料

本项目的原始文献就是工作区根目录中的五份 PDF。Git 仓库保存
`papers_manifest.tsv` 中的 SHA256、页数和书目信息，但不复制 PDF：其中既有
开放许可文献，也有明确标注 Springer rights reserved 的文件，不能默认通过代码
仓库再分发。复核时应对本地文件重新执行 `sha256sum *.pdf`，哈希必须与清单一致。

## 主要设计来源

### Chen、Kelly、Xiu：Expected Returns and Large Language Models

PDF 首页摘要给出的论文事实包括：使用 BERT、RoBERTa、LLaMA、LLaMA2 和
OpenAI 类表示预测新闻后的股票收益；覆盖 16 个全球股票市场和 13 种语言；复杂
叙事与否定语境中 LLM 相对简单 NLP 的优势更明显。论文第 20 页明确使用年度滚动
的 6 年训练、2 年验证、1 年样本外测试，验证期选择参数。

我们直接采用：

- 文本表示与监督预测分离；
- 6+2+1 严格滚动；
- 降维、标准化、聚类和参数选择只能在历史窗口拟合；
- 样本外预测再进入分位数组合和成本回测；
- 正文上下文、新闻时效与复杂叙事是机制变量。

我们主动扩展：

- 从完整文档 embedding 进一步定位 prompt 目标 token embedding；
- 比较双向编码器 RoBERTa/BGE-M3 与后置 prompt 的因果 Qwen embedding；
- 使用新浪新闻和巨潮公告两个中文数据源；
- 增加 mask、PCA、软/硬聚类、UMAP+HDBSCAN 和预测级融合；
- 单独报告新闻同池选优、全市场相对收益与成本后持仓，避免混为一谈。

### Yang：Investor Sentiment in the Chinese Stock Market

该博士论文包含中国市场情绪的短期/长期收益预测、横截面双重排序、
Fama-MacBeth、经济价值、卖空限制，以及 Google Trends/Baidu Index 灾害情绪。
论文第 65 页附近说明其横截面策略在允许卖空假设下进行高情绪减低情绪，并继续
考虑换手和交易成本。

我们据此加入：

- A 股卖空限制、涨跌停、停牌、T+1 和交易成本边界；
- 只做多与理论多空分开评价；
- 未来可将百度指数、投资者关注和股吧情绪作为控制或交互变量；
- 波动率、成交量、流动性和新闻覆盖应作为异质性/风险控制，而非任意 prompt padding。

### Jiang 等：Deep learning, textual sentiment, and financial market

论文以市场收益作为 BERT 情绪训练标签。其正文报告使用 2001--2021 年超过
550 万条媒体新闻；BERT 情绪的月度样本内/样本外预测 R2 为 9.47%/8.39%，
词典情绪对应为 3.18%/0.62%。这些是论文的市场级月度结果，不能与本项目个股日频
RankIC 直接比较。

我们据此保留：

- 收益监督下的文本表示可能优于固定情绪词典；
- 极端期、宏观状态和非线性是后续稳健性方向；
- 任何对比必须注明频率、预测对象和指标，不能把 R2、Accuracy、RankIC 与 Sharpe
  混成一个排名。

### Zhang：Chinese text sentiment mining and stock market correlation

该文使用 2019-01-01 至 2020-12-31 的东方财富深证成指股吧文本，以 BERT 情绪和
最大信息系数研究指数波动。它支持中文社交文本与非线性相关的探索价值，但样本短、
对象为指数，不构成本项目个股新闻预测的直接基准。

### Wang：Topic Modeling in Finance

该综述整理 140 篇、40 本期刊的金融 topic-model 文献，讨论 LDA、sLDA、CTM、
sentence-LDA、STM，以及噪声、标签主观性和黑箱问题。我们把聚类定位为表示结构
诊断和条件因子构造，不以 silhouette/ARI 代替样本外收益证据，正是对这些风险的
直接控制。

## 当前核心假设

1. RoBERTa 的四个目标 token 因子差异较大，可能更多反映 tokenizer/上下文化的
   prompt-specific 变化，而不一定表示模型正确理解了“盈利、收益、亏损”的共同语义。
2. BGE-M3 的四 token 因子更相似，可能表示更平滑的共同语义，但也可能丢失有用的
   细粒度差异。
3. Qwen 若同时表现出接近 BGE-M3 的语义相似度和低于 BGE-M3 的因子重合，并且四
   token 融合稳定超过最佳单 token，才支持“理解共同语义且保留细微差别”。
4. token 是否有效最终由同新闻、同标签、同滚动窗口、同 PCA 算法下相对正文
   `body_mean/article_mean` 的样本外增量决定，不能由 embedding cosine 单独判定。

以上四条仍是假设。三模型四 prompt 公平比较尚未运行完成，交接时不能写成已验证结论。
