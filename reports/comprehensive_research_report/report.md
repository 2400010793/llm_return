<div class="cover" markdown="1">

# 中国市场 Prompt Token 与新闻收益预测研究报告

<div class="subtitle">从全文表示到目标 token：新浪新闻、巨潮公告与三代中文文本模型的严格样本外研究</div>

本报告系统记录 2010--2026 年中国股票新闻与公告的文本表示、Prompt 机制、目标
token、PCA、聚类、收益预测和风险标签研究。研究重点不是寻找单个最高回测值，而是
回答：在相同新闻、相同标签、相同历史窗口和相同降维协议下，目标 token 相对正文
是否提供可重复增量，以及这种增量能否跨模型、Prompt、数据源和年份成立。

<div class="meta">

研究范围：新浪新闻 796,553 条、巨潮公告 903,665 条及旧新浪严格对齐样本 75,894 条<br>
数据截至：新浪 2026 年 8 月 18 日，巨潮 2026 年 7 月 31 日<br>
事实冻结：2026 年 8 月 26 日<br>
报告性质：研究报告，不构成投资建议或可部署策略声明

</div>

</div>

## 目录

[TOC]

<!-- PAGEBREAK -->

## 1. 引言：研究问题与主要发现

新闻全文向量能够压缩公司经营、订单、监管、盈利和风险信息，但全文池化也会把大量
与预测任务无关的语句平均进去。Prompt 提供了一个受控的语义查询：同一篇新闻在
“分析股票收益”“分析股票亏损”等条件下产生不同的上下文化 token。如果目标 token
确实定位了任务相关语义，它应当在严格时间外预测中相对正文产生增量；如果它只反映
词形、长度或模型位置结构，则相似度变化不应被解释为经济理解。

本项目因此沿两条互补路线展开。第一条复现“冻结文本模型加轻量监督预测头”的论文
骨架，建立新闻/公告到未来收益的滚动预测。第二条把研究单位从整篇 pooled embedding
推进到 Prompt 的精确目标 span，比较 RoBERTa、BGE-M3 和 Qwen3-Embedding-8B 的
token、正文及模型间融合。为了减少公司身份记忆和时间泄漏，新增实验只采用
`masked_short`，只 mask 公司名、代码、日期和时间。

本文统一术语如下：RankIC 是每天预测排序与实现目标排序的 Spearman 相关再按日平均；
严格 6+2+1 表示 6 年训练、2 年验证、1 年测试；PCA、标准化、聚类器和监督模型都只能
在当前历史窗口拟合；`body_mean` 是正文 token 平均，`prompt_mean` 是 Prompt 槽位平均，
目标 token 是排除前缀、句号、separator 和 special token 后的精确目标 span。

研究得到七项主要发现。

1. **文本表示中存在跨年的弱收益排序信息。** 新浪 `event_return_3d` 当前最高跨年
   RankIC 为 0.08945，来自 RoBERTa 和 BGE-M3 各自 PCA64 后拼接的 masked“收益”
   span Ridge；新浪 `next_day_return` 当前最高为 0.06044，来自 RoBERTa masked
   `分析股票收益` 的 `prompt_mean` Ridge，均为 9/9 个测试年正。
2. **Prompt 有效，但不存在一个词在所有口径都最好。** 四方向软聚类最高单配置是
   BGE-M3“超额收益”0.04439；跨配置平均由“收益”领先；固定成本后只做多由
   BGE-M3 masked“亏损”领先，净年化 8.30%、Sharpe 0.595。
3. **Token 的模型结构差异是真实的，但不能直接等同于理解能力。** 在 30,946 个共同
   股票日上，RoBERTa 四 Prompt token 因子平均相关约 0.653，BGE-M3 约 0.892；正文
   因子则分别约 0.997 和 0.972。RoBERTa 更 prompt-specific，BGE-M3 更平滑，但哪个
   更有用仍必须由同样本的样本外增量判定。
4. **Qwen token 尚未整体超过正文。** 当前新浪 PCA32 结果中，Qwen masked“收益”
   token RankIC 为 0.05360，低于 masked article mean 的 0.05680；成本后年化 7.37%
   也低于正文 8.57%。局部软聚类更强不足以改写这一总体结论。
5. **聚类主要是结构诊断，不是稳定的主预测器。** 四方向硬 KMeans 相对同 PCA Ridge
   的平均 RankIC 增量为 -0.00026，仅 6/16 胜。UMAP+HDBSCAN 在一个 BGE-M3 loss
   输入上把多空从 15.26bp 提高到 20.35bp，但仍是局部结果。
6. **语义轴必须匹配预测标签。** 估值 token 对下一日 PE 历史相对偏离取得 0.0764
   RankIC，正文为 0.0651；波动率 token 对下一日绝对实现波动率为 0.0553，与正文
   0.0550 几乎相同，对波动率 log 增量则更差。这证明“预测水平”和“预测变化”是不同任务。
7. **当前最关键的结论仍未完成验证。** 三模型、四 Prompt、token/body 的严格共同样本
   PCA32 比较已经设计，但尚未产生完整结果。因此不能写成“Qwen 已经证明兼顾共同语义
   和细粒度差别”，也不能写成“token 普遍优于正文”。

当前证据支持一个审慎判断：**目标 token 是有条件的文本因子候选，其价值依赖模型、
Prompt、标签和监督方法；现阶段最可靠的预测器仍是严格滚动的低维线性模型。**

## 2. 文献基础与本地研究任务

### 2.1 工作区原始论文

本项目的文献依据来自工作区内五份原始 PDF，而不是二手摘要。报告仓库不再分发 PDF，
但 `references/papers_manifest.tsv` 保存文件名、页数、书目信息和 SHA256，可逐份核验。

| 研究 | 原始文件 | 核心方法或问题 | 本项目借鉴 |
|---|---|---|---|
| Chen、Kelly、Xiu，*Expected Returns and Large Language Models* | `ExpectedReturns_LLMs.pdf` | 冻结语言表示、收益监督、16 市场/13 语言、6+2+1 年度滚动 | 主滚动协议、文本表示与预测头分离、样本外组合 |
| Yang，*Investor Sentiment in the Chinese Stock Market* | `Phd_thesis_Zhijiao_Yang_20219273_.pdf` | 中国市场情绪、横截面检验、成本、卖空限制与百度指数 | A 股约束、只做多/理论多空分开、关注度控制 |
| Jiang 等，*Deep learning, textual sentiment, and financial market* | `Deep_learning_textual_sentiment_and_financial_mark (1).pdf` | 以收益标签训练 BERT 情绪，市场级月度预测 | 收益监督文本表示、非线性和状态异质性 |
| Zhang，中文 BERT 情绪与股票相关研究 | `2205.04743v1.pdf` | 2019--2020 东方财富股吧、BERT 与 MIC | 中文语义和非线性相关的探索价值 |
| Wang，*Topic Modeling in Finance* | `134c5fadcdf98503b93ac0ef93b4f0a41b8d.pdf` | 金融 topic model 与聚类综述 | 聚类作为结构诊断，避免用 silhouette 代替收益证据 |

Chen、Kelly、Xiu 的第 20 页明确使用 6 年训练、2 年验证、1 年测试。Jiang 等报告的
2001--2021 年 550 万余条新闻和月度样本外 R² 8.39% 是市场级月度结果，不能与本项目
个股日频 RankIC 直接比较。不同频率、预测对象和评价指标必须分别说明。

### 2.2 从论文骨架到本地研究问题

| 层次 | 本地问题 | 主要比较 | 当前状态 |
|---|---|---|---|
| 全文基线 | 新闻/公告全文是否预测未来收益 | `body_mean/full_mean` + Ridge/Huber/MLP | 新浪、巨潮均已有结果 |
| Prompt 条件 | 加入简短任务词是否改变表示 | short 与 masked-short、Prompt 间配对 | 历史四方向完成 |
| Token 定位 | 精确目标 span 是否优于正文 | token 对 `body_mean/article_mean` | 两模型完成，三模型公平比较待完成 |
| 几何结构 | 模型是否把四个词表示为共同语义 | cosine、centered cosine、CKA、ARI | 部分完成，统一三模型待完成 |
| 语义因子 | 估值、波动率、流动性等是否预测匹配标签 | 三点轴与风险/估值目标 | 2026 单折探索完成 |
| 模型融合 | 多 Prompt、多模型是否产生非线性增量 | Ridge/ElasticNet/HistGB late fusion | 设计和部分旧结果，公平比较待完成 |

### 2.3 本地创新不是机械复刻

原论文研究完整新闻向量，本项目进一步研究 Prompt 目标 token 的机制。新增设计包括：

- 用 `masked_short` 删除公司身份、代码、日期和时间，但保留目标语义词；
- 明确区分 Prompt 前缀、目标 span、正文、separator 和 special token；
- 对 RoBERTa/BGE-M3 使用真正的 `body_mean`，对 Qwen 使用 `article_mean`；
- 双向模型的后置 Prompt 可能影响正文表示，因果 Qwen 的后置 Prompt 不反向影响正文；
- 三模型原始维度 768/1024/4096 不直接共享坐标，只统一算法、PCA 维数和历史窗口；
- 从收益扩展到估值、确定性、波动率、冲击、流动性和风险，但要求标签语义匹配。

## 3. 数据资产与清洗方法

### 3.1 三套不可混用的样本

| 样本 | 期间 | 行数 | 股票数 | 主要用途 |
|---|---|---:|---:|---|
| 新浪全量清洗面板 | 2010-01-01 至 2026-08-18 | 796,553 | 5,466 | 六中性 Prompt 全量 embedding、后续标签研究 |
| 巨潮全量公告面板 | 2010-01-03 至 2026-07-31 | 903,665 | 816 | 全量公告 pooled/Prompt、估值和风险标签 |
| 旧新浪单股对齐面板 | 2010-01-01 至 2026-08-16 | 75,894 | 5,085 | 四方向 Prompt 的历史 RoBERTa/BGE/Qwen 研究 |

三套样本的 `row_index` 都在各自面板内唯一，但含义不同。全量新浪和巨潮不能沿用旧
75,894 条面板的行号。三模型四 Prompt 构建器预期的冻结最小交集为新浪 35,576 条、
巨潮 254,544 条；这是构建时必须核对的预期值，不是已经完成的最终分析样本。

### 3.2 年度分布

| 年份 | 新浪全量 | 巨潮全量 | 旧新浪对齐 |
|---:|---:|---:|---:|
| 2010 | 17,741 | 49,255 | 3,849 |
| 2011 | 20,264 | 50,982 | 4,342 |
| 2012 | 14,658 | 56,101 | 3,635 |
| 2013 | 5,682 | 59,963 | 2,100 |
| 2014 | 1,320 | 71,882 | 448 |
| 2015 | 2,120 | 86,714 | 686 |
| 2016 | 7,797 | 90,227 | 1,463 |
| 2017 | 6,884 | 87,964 | 1,459 |
| 2018 | 33,170 | 39,524 | 5,171 |
| 2019 | 66,774 | 38,070 | 8,097 |
| 2020 | 85,195 | 39,675 | 5,369 |
| 2021 | 82,768 | 41,081 | 3,478 |
| 2022 | 74,865 | 42,669 | 2,969 |
| 2023 | 47,643 | 45,543 | 2,104 |
| 2024 | 60,843 | 39,929 | 2,744 |
| 2025 | 59,885 | 42,219 | 3,524 |
| 2026 | 208,944 | 21,867 | 24,456 |

新浪 2026 年占比明显上升，巨潮 2010--2017 的公告量明显高于 2018 年以后；这类来源
和采集结构漂移必须通过年度滚动、逐年指标和固定新闻池诊断，不能把全期随机切分作为主结果。

### 3.3 文本与标签覆盖

新浪正文清洗字符数中位数为 2,579，90 分位为 9,384，固定最大清洗长度为 12,000 字；
796,553 条中有 646,222 条 next-day 标签和 643,517 条 event-3 标签。巨潮正文字符数
中位数 1,765，90 分位 11,737；882,610 条至少有 100 字，895,251 条有 next-day 标签，
894,851 条有 event-3 标签。

巨潮 903,665 条只覆盖当前面板中的 816 只股票。因此“公告条数大”不等于“横截面更广”。
新浪覆盖 5,466 只股票，但不同年份和来源的有效覆盖不均。报告中的模型比较必须同时冻结
新闻行、股票日、标签可用性和测试年份。

### 3.4 Mask 与点时约束

新增中性 Prompt 只 mask 公司名、股票代码、日期和时间，不删除盈利、亏损、风险、
估值或其他目标词。每行保存稳定 `row_index`、`document_id`、股票代码、发布时间、正文
哈希和 Prompt spec。mask 的价值首先是控制身份与时间泄漏，不是天然的模型增强。

## 4. Embedding、训练与评价流程

### 4.1 模型与表示

| 模型 | 原始维度 | 输入上限 | 正文表示 | 目标 token |
|---|---:|---:|---|---|
| 中文 RoBERTa | 768 | 512 | `body_mean` | Prompt 中精确目标 span 均值 |
| BGE-M3 | 1,024 | 1,000 | `body_mean` | Prompt 中精确目标 span 均值 |
| Qwen3-Embedding-8B | 4,096 | 既有任务合同 | `article_mean` | 排除末尾句号的目标 span 均值 |

六个中性 Prompt 固定为：`分析股票估值`、`分析股票确定性`、`分析股票波动率`、
`分析股票冲击`、`分析股票流动性`、`分析股票风险`。不添加“请”“综合分析”、高中低
或无意义 padding。RoBERTa/BGE-M3 输出同时保存 prompt/title/body/full 的 mean/max、
CLS 和 `prompt_token_embeddings`，因此不是只保存 Prompt token。

### 4.2 四方向 Prompt 的 span 定义

| Prompt | 目标 span | 说明 |
|---|---|---|
| `分析股票盈利` | `盈利` | 两个中文 token 的均值 |
| `分析股票收益` | `收益` | 两个中文 token 的均值 |
| `分析股票超额收益` | `超额收益` | 公平比较使用完整四字目标 span；部分历史软聚类使用“超额”并单列 |
| `分析股票亏损` | `亏损` | 两个中文 token 的均值 |

历史结果保留原有 span 定义，不能事后更改。新的三模型公平比较使用完整目标短语，并在
manifest 中保存 input IDs、tokens、offset mapping、目标 token 索引和 Prompt hash。

### 4.3 时间切分与固定参数

新浪主结果使用严格 6+2+1，测试年 2018--2026。巨潮三模型公平比较因历史 embedding
交集限制，基础预测采用 3 年历史加 1 年测试，并在历史窗口内保留 1 年 OOS 供二级融合。
语义轴与高频标签的现有探索使用 2018--2023 训练、2024--2025 验证、2026 测试；高频
标签 2018--2020 为空，因此实际有效训练年为 2021--2023。

当前公平比较预注册参数为 PCA32、`random_state=42`、Ridge alpha 100、KMeans k=6、
seeds 17/29/42/71/113、soft shrinkage 500、temperature 0.5、UMAP8 和 HDBSCAN
`min_cluster_size=50, min_samples=10`。不做原始维回归，也不根据九年最终结果改参数。

### 4.4 评价指标

主指标为日均 RankIC、RankIC IR、逐年 RankIC、正 RankIC 年份数、月度正 RankIC 比例
和日期区块 bootstrap 区间。辅助指标为同新闻池 Top20% 超额、多空收益、Accuracy、
样本外 R²、成本后年化/Sharpe/回撤、因子 Spearman 相关和 Top20% 选股重合率。

Embedding cosine、centered cosine、CKA、ARI 和 silhouette 只描述表示结构，不能选择
收益模型。三模型原始坐标不可直接比较；跨模型只比较标准化几何统计和完全样本外因子。

### 4.5 组合与成本

四方向历史成本回测固定 `gamma=0.1`，买入和卖出各 5bp。只做多、独立做空和多空必须
分别报告。A 股缺少逐股券源、借券费和召回数据时，任何空头腿都是理论诊断，不能称为
可部署策略。`simple_states` 返回 `ErrorCode=1002` 的结果只作为统一平台对照，不替代
正式成本回测。

## 5. 新浪四方向 Prompt 方法验证

### 5.1 完成范围

旧新浪 75,894 条母面板上，四个 Prompt、两个模型、short/masked-short 两个变体、
九个测试年共 144 个独立软聚类 fold，已完成 144/144。每个 fold 的 PCA 维度、簇数、
收益响应、收缩和温度只由验证期选择，测试前再用训练加验证拟合。

### 5.2 样本外排序

| 配置 | 日均 RankIC | 正年份 | 正月份 |
|---|---:|---:|---:|
| 超额收益 / BGE-M3 / short / RankIC 选择 | 0.04439 | 9/9 | 72/104 |
| 收益 / BGE-M3 / short / RankIC 选择 | 0.04327 | 9/9 | 70/104 |
| 亏损 / RoBERTa / short / RankIC 选择 | 0.04317 | 8/9 | 65/104 |
| 收益 / RoBERTa / short / RankIC 选择 | 0.04061 | 9/9 | 74/104 |
| 盈利 / RoBERTa / masked-short / RankIC 选择 | 0.03920 | 9/9 | 70/104 |

<div class="figure">
<img src="figures/direction_prompt_rankic.png" alt="Direction prompt RankIC">
<p>图 1. 四方向 Prompt 软聚类的代表性样本外 RankIC。图中配置不是同一 Prompt 的因果实验。</p>
</div>

四个 Prompt 各自八种配置平均后，“收益”RankIC 为 0.03529，“亏损”0.03164，
“超额收益”0.03034，“盈利”0.02995。最高单配置与跨配置平均不是同一问题。

### 5.3 无聚类线性基线

无聚类 Ridge 中，四 Prompt 各八种配置平均 RankIC 为：盈利 0.04480、收益 0.04393、
超额收益 0.04385、亏损 0.04281，最大平均差仅 0.00199。单配置最高为 RoBERTa masked
“收益”`prompt_mean` 的 0.06044。四词都含排序信息，但没有证据表明某个词义具有稳定
因果优势。

### 5.4 成本后的不同答案

| 目标 | 最佳配置 | 净年化 | 净 Sharpe | 最大回撤 |
|---|---|---:|---:|---:|
| 只做多 | 亏损 / BGE-M3 masked / token 软聚类 | 8.30% | 0.595 | -22.87% |
| 理论多空 | 亏损 / RoBERTa short / token 软聚类 | 4.64% | 0.515 | -26.45% |

所有独立 `short_only` 配置的成本后 Sharpe 均为负。现有经济结果的主要含义是“低分组
相对更弱”和持仓平滑可能有价值，不能直接解释为可实施卖空收益。

### 5.5 Mask 的配对结果

16 个严格配对配置中，mask 对只做多净 Sharpe 的平均增量为 +0.069，9/16 胜；对
`simple_states` IC 的平均增量却为 -0.00480，仅 8/16 胜。mask 可能通过改变持仓尾部
改善经济指标，但没有稳定提高排序。身份控制与预测增强必须分开表述。

## 6. Token、正文与模型几何

### 6.1 两模型四 Prompt 横向结构

现有比较使用新浪 30,946 个完全共同股票日上的滚动样本外因子，不是直接把 768 维和
1,024 维原始向量放在同一坐标系计算距离。

| 指标 | RoBERTa | BGE-M3 |
|---|---:|---:|
| 四 Prompt 正文因子平均 Spearman | 0.997 | 0.972 |
| 四 Prompt token 因子平均 Spearman | 0.653 | 0.892 |
| token 与对应正文 Top20 重合 | 44% | 70% |
| masked 正文平均 RankIC | 0.04391 | 0.03307 |
| masked 目标 span 平均 RankIC | 0.04010 | 0.03154 |

RoBERTa token 的 Prompt 间差异更大，BGE-M3 token 更接近共同方向。两种解释都可能
成立：前者可能保留有用细粒度，也可能只是词形敏感；后者可能理解共同语义，也可能过度
平滑。只有同新闻、同标签、同 PCA 的 token/body 增量和多 Prompt 融合才能区分。

### 6.2 Qwen 的当前反例

| 表示 | PCA32 RankIC | 成本后年化 |
|---|---:|---:|
| masked article mean | 0.05680 | 8.57% |
| masked“收益”token | 0.05360 | 7.37% |

<div class="figure">
<img src="figures/token_body_rankic.png" alt="Token and body RankIC">
<p>图 2. 三模型现有 token/body 结果。RoBERTa/BGE-M3 是四 Prompt 汇总，Qwen 是单一“收益”Prompt，不能当作完全公平的模型排名。</p>
</div>

Qwen 当前 token 没有整体超过正文。Qwen 是因果模型且 Prompt 位于正文之后，其
`article_mean` 不受后置 Prompt 反向影响；RoBERTa/BGE-M3 为双向编码器，Prompt 可能
改变正文位置的表示。公平比较必须记录这种模型结构差异。

### 6.3 三模型公平比较的判定门槛

只有同时满足以下条件，才支持“Qwen 兼顾共同语义和细微差别”：RoBERTa centered
cosine 显著低于 BGE-M3；BGE-M3 因子相关和选股重合最高；Qwen 几何接近 BGE-M3
但因子重合更低；Qwen 四 token 融合超过最佳单 token 和 Qwen 正文；融合增量至少
三分之二测试年为正且区块 bootstrap 区间不跨 0；三模型融合继续超过最佳单模型。

该比较尚未完成。当前只能报告两模型结构事实和 Qwen 单 Prompt token/body 结果。

## 7. 中性语义轴、估值与风险标签

### 7.1 从方向词扩展到任务词

研究先构造短/中/长收益、低/中/高估值、确定性、波动率、冲击和流动性三点轴，再形成
direction、high/low deviation、magnitude、clarity 和 neutral distance。后续又提交六个
不含高中低的中性 Prompt，用于判断“高/低”字符是否真正改变表示。

中性 Prompt 不是长度 padding。所有比较保留原始短词，并在同一模型内部使用同一
tokenizer、最大长度、mask 和正文截断合同。不同目标词 token 数不同时，不直接比较原始
token 坐标，而比较训练期标准化后的因子和样本外预测。

### 7.2 收益标签上的单年结构探索

18 个新 Prompt 的 RoBERTa masked-short token direction 与 body direction 在 2026
严格测试上的平均配对结果如下。

| 方法 | 平均 Token-Body RankIC | RankIC 胜出轴数 | 解释 |
|---|---:|---:|---|
| PCA32 + Ridge | +0.03670 | 4/6 | 线性主结果在部分轴显示 token 增量 |
| PCA + hard KMeans + Ridge | +0.03344 | 5/6 | 硬簇特征局部有效 |
| soft KMeans | -0.01451 | 1/6 | 不稳定 |
| UMAP8 + HDBSCAN + Ridge | -0.00881 | 3/6 | 平均无增量 |

这是 2026 单个测试年，不能替代至少 6/9 年稳定性或日期区块 bootstrap。它支持
“新 token 改变了可预测表示”，不支持“所有语义轴 token 普遍优于正文”。

### 7.3 估值的可检验定义

高估/低估不能由 PE 的绝对值跨股票直接比较。本项目把 PE、PB、PS 和 EV/EBITDA 的
正值取对数，再减去该股票过去 252 个交易日、至少 60 日的滞后 log 中位数，得到历史
相对偏离；预测标签使用下一交易日的该偏离。这控制了不同股票长期估值基准差异。

| 标签 | Token RankIC | Body RankIC | Token-Body | 2026 样本 |
|---|---:|---:|---:|---:|
| 下一日 PE log 偏离 | 0.0764 | 0.0651 | +0.0113 | 1,119 |
| 下一日 PB log 偏离 | -0.0056 | 0.0197 | -0.0254 | 1,683 |
| 下一日 PS log 偏离 | 0.0106 | -0.0069 | +0.0175 | 1,683 |
| 下一日 EV/EBITDA log 偏离 | 0.0107 | 0.0059 | +0.0048 | 1,274 |

PE 的点估计较好，但不同估值指标方向不一致，且样本只有约 1,100--1,700 个股票日。
不能合并写成“估值 Prompt 已被证明有效”。

### 7.4 波动率和流动性的三个层次

波动率实验区分：下一交易日实现波动率绝对水平；下一日相对前一日的 `log1p` 变化；
五日后相对二十日前的波动率跳升。前者容易包含股票固有波动率差异，第二个更接近增量，
第三个检验中期风险变化。

| 标签 | Token RankIC | Body RankIC | Token-Body | 2026 样本 |
|---|---:|---:|---:|---:|
| 下一日实现波动率水平 | 0.0553 | 0.0550 | +0.0003 | 1,671 |
| 下一日实现波动率 log 变化 | -0.0236 | -0.0085 | -0.0151 | 1,671 |
| 五日/二十日波动率跳升 | 0.0310 | 0.0253 | +0.0057 | 1,674 |

流动性使用一分钟买卖价差聚合。下一日价差水平 token/body RankIC 为 0.1881/0.2151；
价差 log 变化为 -0.0092/-0.0325。高水平预测强，但可能主要反映稳定的个股流动性差异；
变化预测较弱。因此下一步应增加逐股/行业残差化和滞后标签基线。

## 8. 巨潮公告扩展

### 8.1 全量 pooled 结果

巨潮全量 pooled embedding 使用 903,665 条公告、RoBERTa/BGE-M3、short 和
masked-short，共 72 个一级样本外回归配置，覆盖 2018--2026。

| 配置 | 平均 RankIC | 正年份 |
|---|---:|---:|
| RoBERTa masked full_mean SmallMLP PCA128 | 0.01764 | 9/9 |
| RoBERTa short body_mean SmallMLP raw | 0.01665 | 7/9 |
| BGE-M3 masked concat Huber raw | 0.01415 | 9/9 |
| BGE-M3 masked full_mean Huber raw | 0.01369 | 9/9 |

这些结果说明公告 pooled 表示存在较弱但跨年为正的信号，也说明不同模型的最优预测头
不同。它们不是四 Prompt token 的公平比较，不能拿来回答 token 是否优于正文。

### 8.2 Qwen 巨潮结果

Qwen 巨潮 3+1+1 辅助滚动的 masked-short PCA32 Huber SGD RankIC 为 0.02732，5/5
年方向一致；严格 2026 点估计为 0.02247，但 95% 区间 `[-0.00181, 0.05117]` 跨 0，
且 plain 的点估计略高于 masked-short。当前只能称为弱而可重复的线性信号。

### 8.3 横截面和数据源边界

巨潮全量有 90 万余条公告，却只覆盖面板中的 816 只股票；新浪股票更广但来源、长度和
年份结构不同。两数据源的模型结果不能按原始 RankIC 直接归因于“新闻优于公告”或反之。
公平比较需要同一股票日、同一目标、相同训练历史和相同可评分横截面。

## 9. 排序、收益与风险解释

### 9.1 IC 与组合收益不是同一指标

RankIC 使用每天整个可评分横截面的排序；Top20% 组合只看尾部，并受持仓平滑、换手、
成本和收益偏度影响。因此“超额收益”有最高软聚类 RankIC，而“亏损”有最高成本后只做多，
并不矛盾。报告必须同时给出排序和经济结果。

### 9.2 Token 的价值可能集中在尾部

旧比较中，RoBERTa token 与正文 Top20 重合只有约 44%，BGE-M3 约 70%。RoBERTa
盈利、收益、亏损 token 相对对应正文的多空收益增量曾为正，而超额收益是反例。这提示
token 可能主要改变尾部选股，而不是整体 RankIC；但该机制需要在新公平样本上复核。

### 9.3 成本和卖空限制

四方向最佳只做多年化 8.30%、Sharpe 0.595，最大回撤仍为 -22.87%。最佳理论多空
年化 4.64%、Sharpe 0.515，且独立空头配置成本后均未形成正 Sharpe。没有券源与借券费
时，应把低分端理解为风险规避或降低权重候选，而不是直接做空策略。

### 9.4 需要补充的风险归因

下一步的经济评价应在同一股票日加入市场和行业调整收益、规模、动量、短期反转、beta、
历史波动率、流动性、换手和新闻数量。只有“传统因子”“新闻因子”“传统加新闻”在完全
同折下比较，才能判断文本是否提供独立增量。

## 10. PCA、聚类与模型融合

### 10.1 为什么主结果只做 PCA 后回归

三模型原始维度差异大，且高维样本量比例不同。主口径固定训练期 randomized PCA32 后
StandardScaler 和 Ridge alpha 100，不再运行原始维回归。这使跨模型统一成为“同算法、
同维数、同历史窗口和同参数”，而不是错误地共享同一个 PCA 基底。

### 10.2 聚类的正面和负面证据

硬 KMeans 在四方向 16 个严格配对中平均降低 RankIC 0.00026，仅 6/16 胜。BGE-M3
收益 token 的簇收益排序比 RoBERTa 清晰，说明其表示更容易形成连续分层；但软聚类最高
0.04439 仍低于无聚类 Ridge 0.06044。

UMAP+HDBSCAN 在 BGE-M3 masked loss 的同一输入上把多头从 1.50bp 提高到 6.17bp，
多空从 15.26bp 提高到 20.35bp，8/9 年方向为正。它是值得保留的局部稳健性结果，
但没有跨 Prompt 稳定胜出，也没有超过已有最佳软聚类配置。

### 10.3 多因子融合的严格方案

公平比较先生成 24 个完全样本外基础因子：3 模型 × 4 Prompt × token/body。融合分为
单模型四 token、单 Prompt 三模型、12 token、12 body 和全部 24 因子。输入先逐日
横截面标准化，再比较等权、Ridge、ElasticNet 和 HistGradientBoosting。二级权重只能
使用历史 OOS/验证数据，测试年冻结。

### 10.4 当前不能写出的结论

- 不能因 RoBERTa Prompt 间距离大就写“RoBERTa 不理解语义”；
- 不能因 BGE-M3 token 相关高就写“BGE-M3 理解更准确”；
- 不能因 Qwen 局部软聚类更强就写“Qwen token 已超过正文”；
- 不能把新浪 35,576 和巨潮 254,544 的预期交集当作已完成样本；
- 不能用一次 2026 单折的估值/风险结果替代 6+2+1 跨年验证。

## 11. 综合判断与下一阶段

### 11.1 当前证据等级

| 结论 | 证据 | 当前判断 |
|---|---|---|
| 文本 embedding 含收益排序信息 | 新浪多个配置 9/9 年正；巨潮 pooled 多个配置跨年正 | 支持弱文本因子 |
| 简短 Prompt 改变可预测表示 | 四方向结果、mask 配对、不同 span 排名 | 支持，但非稳定增强 |
| Token 普遍优于正文 | 两模型平均 RankIC token 略低；Qwen token 也低于正文 | 不支持 |
| RoBERTa 与 BGE 的 Prompt 几何不同 | 因子相关 0.653 对 0.892，正文相关均很高 | 支持结构差异，不等于理解优劣 |
| 聚类稳定提高收益预测 | 硬聚类平均负增量；UMAP 只有局部正例 | 当前不支持 |
| 估值 Prompt 预测高估/低估 | PE 有正点估计，PB/PS/EV 分化，单年小样本 | 探索性证据 |
| 波动率 Prompt 预测未来风险 | 水平略正，log 增量更弱 | 仅支持水平信息，增量未验证 |
| Qwen 兼顾共同语义和细粒度并可融合 | 严格三模型四 Prompt 结果未完成 | 尚未验证 |

### 11.2 当前可保留的基线

1. 新浪 next-day：RoBERTa masked `分析股票收益` prompt_mean PCA/Ridge 主基线。
2. 新浪 event-3：RoBERTa/BGE-M3 masked“收益”span 各自 PCA 后拼接 Ridge。
3. 方向 Prompt 机制：四词目标 span token 与真正 `body_mean` 的严格配对。
4. 巨潮公告：现有 pooled 72 配置只作为公告全文基线；三模型 token 比较单独报告。
5. 非线性：只在 PCA32 因子或完全 OOS 预测层运行 HistGradientBoosting，不再扩展高维树网格。

### 11.3 下一阶段优先级

1. 完成新浪和巨潮三模型四 Prompt 共同样本构建，先核对实际交集、年份、标签和选择偏差。
2. 运行 PCA32 token/body Ridge，并生成逐年 RankIC、配对 bootstrap、因子相关和 Top20 重合。
3. 在基础 OOS 因子完成后做等权、Ridge、ElasticNet 和 HistGradientBoosting late fusion。
4. 将估值标签扩展为行业相对和横截面分位；将波动率/价差同时报告水平、变化和滞后基线增量。
5. 把文本因子与量价、beta、行业、规模、动量、反转和流动性因子放入完全共同股票日比较。
6. 只有达到至少 6/9 年方向一致且区块 bootstrap 区间不跨 0，才升级为主结论。

### 11.4 当前资产完整性

六个中性 masked-short Prompt 的 embedding 完整性如下。RoBERTa 新浪完整；巨潮缺
一个分片；BGE-M3 剩余任务已主动取消并保留完成产物，因此只能在已完成分片交集探索。

<div class="figure">
<img src="figures/neutral_embedding_completion.png" alt="Neutral prompt embedding completion">
<p>图 3. 中性六 Prompt embedding 分片完成度。未完成 BGE-M3 不能冒充全量结果。</p>
</div>

## 12. 研究边界

- 新浪与巨潮虽然都覆盖 2010--2026，但年度分布、股票覆盖、文本类型和标签可用率不同。
- 旧新浪 75,894 条、全量新浪 796,553 条和巨潮 903,665 条的 `row_index` 不可混用。
- 2026 新浪数据量异常高，任何随机切分都会受到来源与时间漂移影响。
- 巨潮当前全量面板只覆盖 816 只股票，不代表无偏全 A 股公告历史。
- RoBERTa/BGE-M3 是双向编码器，Qwen 是因果模型；Prompt 位置影响机制不同。
- Qwen 当前只完成单一“收益”Prompt 的 token/body 对照，不能代表四 Prompt 平均。
- 估值、波动率和价差结果目前主要来自 2026 单折和约 1,100--1,700 个股票日。
- 聚类超参数、PCA 和 scaler 必须在训练期拟合；ARI/silhouette 不能选择收益模型。
- 成本回测未包含容量、冲击、涨跌停、融资、借券费和券源召回。
- `simple_states ErrorCode=1002` 的结果不是正式成本后策略证据。
- 本报告描述预测关系和表示结构，不提供因果解释，也不构成投资建议。

<!-- PAGEBREAK -->

## 附录 A：Prompt 与表示合同

| 实验族 | Prompt | 变体 | 目标 span | 正文表示 |
|---|---|---|---|---|
| 四方向 | 分析股票盈利 | masked-short | 盈利 | body_mean/article_mean |
| 四方向 | 分析股票收益 | masked-short | 收益 | body_mean/article_mean |
| 四方向 | 分析股票超额收益 | masked-short | 超额收益 | body_mean/article_mean |
| 四方向 | 分析股票亏损 | masked-short | 亏损 | body_mean/article_mean |
| 中性估值 | 分析股票估值 | masked-short | 估值 | body_mean |
| 中性确定性 | 分析股票确定性 | masked-short | 确定性 | body_mean |
| 中性波动率 | 分析股票波动率 | masked-short | 波动率 | body_mean |
| 中性冲击 | 分析股票冲击 | masked-short | 冲击 | body_mean |
| 中性流动性 | 分析股票流动性 | masked-short | 流动性 | body_mean |
| 中性风险 | 分析股票风险 | masked-short | 风险 | body_mean |

## 附录 B：主要事实来源

| 事实 | 可追溯产物 |
|---|---|
| 全量面板行数、日期、股票数、标签覆盖 | `facts.json`；`scripts/audit_research_handoff.py`；相应 Parquet schema |
| 四方向 Prompt、mask、成本和聚类结果 | 工作区 `REPORT_ALL_RESULTS.md`，SHA256 `23ade165...e4ff` |
| 估值、波动率、价差配对结果 | `reports/aligned_extended_regression/token_body_selected.csv` |
| 新语义轴聚类配对 | `reports/aligned_factor_clusters/token_minus_body_delta.csv` |
| 中性 embedding 完成度 | `docs/status_snapshot_2026-08-26.json` 和 embedding shard `COMPLETED` |
| 论文书目信息和原文校验 | `references/papers_manifest.tsv`、`references/paper_to_experiment_map.md` |

## 附录 C：结果解释的优先级

1. 冻结输入、严格滚动、跨年完成且有明确标签的样本外结果；
2. 同新闻、同标签、同历史窗口的配对结果；
3. 单年严格测试和小样本语义匹配探索；
4. embedding 几何、聚类质量和可视化；
5. 尚未运行完成的设计、预期交集和研究假设。

低优先级证据不能覆盖高优先级证据。尤其不能用 cosine 解释替代 token/body 的样本外
预测，也不能用单个最优组合覆盖所有其他 Prompt 和模型的负结果。

## 附录 F：指标和方法速查

| 名称 | 定义 | 本报告解释 |
|---|---|---|
| 日均 RankIC | 每日预测与目标横截面的 Spearman 相关再平均 | 首要排序指标，不年化 |
| RankIC IR | 日度 RankIC 均值除以标准差 | 衡量稳定性，需同时报告日期数 |
| 正年份数 | 测试年份中 RankIC 大于 0 的年份数 | 方向稳定性，不代表统计显著 |
| OOS R² | 相对历史均值预测误差的改善 | 可为负而 RankIC 为正 |
| Top20 重合 | token 与正文最高 20% 股票集合的交集比例 | 描述尾部选择相似度 |
| centered cosine | 去除训练期 Prompt 均值后的余弦 | 减少全局 Prompt 偏置 |
| 线性 CKA | 两组表示样本几何的一致性 | 可跨不同原始维度描述，不共享坐标 |
| PCA32 Ridge | 训练期 PCA32、标准化、固定 alpha 100 | 三模型公平比较主基线 |
| hard KMeans | PCA 后簇 one-hot 与连续特征进入 Ridge | 聚类增强，必须配对同 PCA Ridge |
| soft KMeans | 距离软分配加训练期簇收益收缩 | 连续分层，不使用测试收益定义簇分数 |
| UMAP+HDBSCAN | 训练期 PCA/UMAP/密度簇后进入 Ridge | 稳健性复核，不以 silhouette 选收益模型 |
| 净年化/Sharpe | 扣固定交易成本后的组合收益统计 | 仍未包含容量、涨跌停和借券约束 |

## 参考文献

<div class="references" markdown="1">

1. Chen, Y., Kelly, B., and Xiu, D. *Expected Returns and Large Language Models*. 2024.
2. Yang, Z. *Investor Sentiment in the Chinese Stock Market*. PhD thesis, 2023.
3. Jiang, F., Liu, Y., Meng, L., and Zhang, H. *Deep learning, textual sentiment, and financial market*. 2024.
4. Zhang, C. *Deep learning based Chinese text sentiment mining and stock market correlation research*. 2022.
5. Wang, X. *Topic Modeling in Finance: A Review of Methods, Applications, and Challenges*. 2026.

</div>
