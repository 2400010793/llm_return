# 中国市场 Prompt Token 新闻收益预测

本仓库研究中文财经文本能否预测 A 股收益，以及 prompt 目标 token 是否相对新闻正文
提供增量信息。项目已形成新浪新闻和巨潮公告双数据源、RoBERTa/BGE-M3/Qwen
三模型、严格滚动 PCA 回归、聚类、组合和交易成本评估的完整研究链。

当前事实快照为 **2026-08-26**。最详细的状态、路径、失败结果和接手命令见
[完整交接文档](docs/HANDOFF_2026-08-26.md)，研究结论见
[综合研究报告](reports/comprehensive_research_report/report.md)，论文依据见
[论文到实验的映射](references/paper_to_experiment_map.md)。

## 1. 当前结论

- 新浪完整清洗面板有 **796,553** 条新闻；巨潮合并面板有 **903,665** 条公告记录。
  二者都覆盖 2010--2026，但巨潮 2010--2017 为 816 只股票的全部公告，2018--2026
  为同一批 816 只股票的重点公告；股票池、文本类型和筛选口径不同，不能混成一个面板。
- 现有新浪四 prompt 结果说明目标 token 确实包含收益排序信息，但 token 并未普遍超过
  正文。Qwen 的“收益”token RankIC 为 0.05360，低于正文 0.05680。
- RoBERTa 的四 prompt token 因子相关较低（平均 0.653），BGE-M3 较高（0.892）。
  这只是模型内几何事实，不能单独证明哪个模型“理解得更好”。
- 新浪当前最佳 event-3 结果是 RoBERTa+BGE-M3 的 masked“收益”span，
  PCA64 后拼接并用 Ridge，RankIC **0.08945，9/9 年为正**。
- 巨潮既有 pooled 结果的最佳 RankIC 为 **0.01764，9/9 年为正**；公告上的信号
  明显弱于新浪新闻，但跨年更稳定。
- 硬 KMeans 平均没有提升线性基线；软聚类和 UMAP+HDBSCAN 只在部分 prompt、
  部分组合尾部有效，必须和同一 PCA+Ridge 基线配对报告。

## 2. 数据来源与采集

| 数据 | 来源与用途 | 当前规模 | 采集边界 |
|---|---|---:|---|
| 新浪财经 | 公开历史财经新闻，正文语义与收益预测 | 796,553 条清洗新闻 | **网络采集只能在本地机器运行**；集群只处理复制并校验后的文件 |
| 巨潮资讯 CNINFO | 官方上市公司公告、详情页和公告 PDF | 903,665 条合并面板记录，816 只股票；2010--2017 全部公告，2018--2026 重点公告 | 只读取公开可见页面和链接，不调用隐藏接口，不绕过验证码或访问控制 |
| 东方财富 | 要闻、个股新闻/公告/研报、股吧可见内容 | 有界批次采集 | 低频浏览器流程；遇访问控制立即停止 |
| 雪球 | 公开可见讨论、资讯和公告 feed | 有界批次采集 | 匿名访问或用户显式提供的授权 browser state；不导出或绕过登录 |
| 行情与高频指标 | 日频 OHLC、收益标签；1 分钟波动率、价差、流动性和估值数据 | 按研究交集对齐 | 原始授权数据不进入 Git，以 manifest、路径和哈希追踪 |

### 2.1 新浪的两种流程

新浪不是一条爬虫完成所有任务，而是两种互补流程。

**流程 A：普通 HTTP 图扩展，适合批量正文。**

入口为 `scripts/run_sina_auto_pipeline.py`，底层为
`scripts/crawl_sina_bfs_fast.py`。当前默认先从种子做 depth-3 抓取，按 URL 和正文
哈希全局去重并排除该 runner 中的 2000 年记录，再把去重后的文章作为 depth-5 根节点
继续扩展，最后再次去重。每页保存原始 HTML、JSONL record、manifest 和可恢复 state。

当前自动流程配置为 `workers=20`、`pause=1s`、`timeout=30s`、正文
`120--12000` 字符；worker 范围限制为 1--50。20 是并发上限，不等于 20 页/秒，
请求、解析、robots 和批次间等待都会限制吞吐。

```powershell
$env:PYTHONPATH = "scripts"
python scripts/run_sina_auto_pipeline.py `
  --root-file data/interim/sina_roots.txt `
  --stock-catalog data/stock_universe_csi500_current.csv `
  --output-dir data/interim/sina_auto_v1 `
  --workers 20 --pause-seconds 1 --timeout 30
```

**流程 B：浏览器渲染发现种子，适合动态页和缺失年份。**

`scripts/select_sina_historical_seeds_browser.py` 从公开渲染页面发现和检查候选，默认
`workers=8`、`pause=3s`、`timeout=30s`、`render_wait=500ms`、depth 2、最多
300 页。它按股票、年份、季度、频道路径和标题差异做 farthest-first 选择，默认每只
股票每年最多一个 seed。`scripts/run_sina_annual_expansion.py` 可按年份独立扩展，默认
4 个年份任务 x 每年 2 个浏览器页面，理论浏览器并发上限 8，depth 5、每年最多
1000 页、目标 100 个 seed。

**推荐选择：混合流程。** 浏览器只负责发现/验证少量历史种子，普通 HTTP 负责正文
扩展和断点恢复；只有动态页面或缺失年份才用全浏览器补洞。普通 HTTP 更快、更容易
审计，浏览器覆盖更灵活但成本高。仓库只有微型探针：普通 DFS 2 页/0.38 秒，浏览器
BFS 4 页/4.90 秒，浏览器 DFS 4 页/4.77 秒。这些样本太小，只用于证明流程可运行，
**不能外推全量速度**。目前没有一次完整新流水线的吞吐报告。

> 硬约束：新浪所有联网发现和抓取命令只在本地 Windows/工作站执行，不提交到
> Slurm，也不在 CPU/GPU 节点抓取。抓完后再将原始 HTML、JSONL、manifest 和 state
> 一起复制到 Lustre，并校验 SHA256。

### 2.2 巨潮公告流程

巨潮是官方公告来源，和新浪新闻不同。`scripts/collect_cninfo_announcements.py` 的流程为：

1. 从公开公司披露页按股票和日期筛选公告索引；
2. 遍历可见分页并保存标题、公告日期和详情 URL；
3. 顺序读取详情页，按公开 PDF 链接下载原文件；
4. 用 `pypdf` 提取文本，保留 PDF、详情文本、采集错误和状态；
5. 按 `document_id` 去重、清洗、分类，并与收益/O2O 面板对齐。

巨潮整体覆盖 2010--2026 和同一批 816 只股票，但筛选规则分段：2010--2017 是全部
公告，历史采集审计覆盖 816 x 8 = 6,528 个股票年且全部完成，原始记录 554,451 条，
清洗面板 553,088 行；2018--2026 是重点公告，清洗面板 350,577 行。两段合并后的最终
分类面板为 903,665 行，日期为 2010-01-03 至 2026-07-31，`row_index` 和
`document_id` 均唯一。正文至少 100 字的记录为 882,610 条，next-day 标签 895,251 条，
event-3 标签 894,851 条。文件名中的 `full` 只表示合并面板完整，不表示每一年都是全部公告。

巨潮的 816 只不是网站爬取上限，而是当前研究范围：采集器接受
`CODE:ORG_ID:NAME`，技术上可处理任何能在巨潮公开公司页解析出 `stockCode/orgId` 的
上市公司。本次完整历史审计使用 1,000 只候选股票与公告/收益面板的交集，实际为沪市 64
只、深市 752 只，未纳入北交所。若扩展到全 A 股，必须先建立新的代码和 orgId 目录，分
股票年探针并做缺失审计，不能直接把 `--expected-stocks` 改成全市场。

新浪的本地限制是访问上下文和可复现性要求，不是 CPU 限制。浏览器发现 seed、HTTP 请求、
历史编码处理、限速、原始文件和 state 都在本地完成；集群只处理复制并校验后的文件。
完整逻辑为：浏览器按股票/年份/季度发现 seed，普通 HTTP 做 depth-3 扩展，按 URL 和
正文 SHA256 去重，再把有效文章做 depth-5 扩展；动态入口和年份断档再用年度浏览器
runner 补洞，最后生成 records/state/manifest/checksum。默认 HTTP 为 20 workers、1 秒
批次 pause、30 秒 timeout；浏览器 selector 为 8 pages、3 秒 pause、500ms render wait。
20 workers 不是 20 页/秒，微型探针不能代表全量速度。禁止在 Slurm CPU/GPU 节点联网抓取。

### 2.3 东方财富和雪球

`scripts/collect_browser_visible.py` 只读取浏览器可见内容：东方财富要闻、个股页、公告、
研报和股吧，以及雪球公开讨论/资讯/公告。默认页面间隔 3 秒。批处理入口默认每批 3 只
股票、批间 60 秒；`run_100_stock_collection.py` 当前历史默认期望 90 只 active 股票，
每次只跑一个 batch，股吧详情 10 篇、个股详情 6 篇、个股链接 20 条、研报详情 0 篇。

这套流程适合投资者情绪、关注度和事件对照，不适合作为高速历史新闻主库。若需要登录
态，storage state 必须由用户显式提供且已获授权；检测到验证码、访问异常或请求过频时
程序会停止，不能增加重试、代理或隐藏 API 绕过。

### 2.4 数据清洗和日期对齐

新浪原始文件不是一行等于一只股票。清洗先解析标题、正文和发布时间，删除页面导航、
分享按钮、编辑/推广尾注等网页模板，过滤无效时间、日期冲突、乱码和少于 120 字的正文；
股票归属必须由清洗后标题/正文中的明确代码或无歧义公司名确认，不能只信页面自带股票
链接。714,555 个输入行中有 714,549 个可解析行，最终接受 167,713 篇源文章；一篇文章
可以对应多只股票，展开后形成 796,553 个股票-文章研究行，覆盖 5,466 只股票。去重依次
使用 `source_article_id+stock_id`、`url+stock_id` 和
`stock_id+cleaned_text_hash`，因此“源文章数”不能与“研究面板行数”混写。

巨潮先从 PDF/详情提取文本，清除页码、HTML、URL、浏览器/PDF 样板和异常空白，再按
`stock_id+text_hash` 去重；相同公告若属于不同股票仍保留。通用合并清洗档从 913,014
行删除 9,198 个重复后得到 903,816 行，但本研究冻结输入使用两个已审计时期包：
2010--2017 全部公告 553,088 行，加 2018--2026 重点公告 350,577 行，共 **903,665**
行。两者相差 151 行是版本/选择合同差异，尚未完成逐行原因审计，禁止用 903,816 中间档
替换 903,665 冻结面板。

日期统一使用交易所日历而不是自然日平移。股票代码先规范为六位，5/6 开头映射 `.SH`，
其余映射 `.SZ`。新浪时间按 UTC 解析后转为 `Asia/Shanghai`；巨潮优先使用
`published_at`，缺失时退回 `announcement_date`，历史公告通常只有日期精度。主面板的
`event_center_date` 是公告日当日或之后的首个交易日，`entry_date` 是**严格晚于公告日**
的首个交易日；周末和节假日自动顺延。`next_day_return` 取 `entry_date` 的日收益，O2O
取 `entry_date` 开盘到下一个交易日开盘。该保守合同不按盘中发布时间切收盘线，避免
日期级公告被误当成盘前可交易信息。估值只允许 backward as-of 合并
`available_at <= published_at`；所有 PCA、标准化和回归仍只在历史训练窗拟合。

## 3. 模型、来源与同口径收益结果

| 模型 | 代码中的模型来源 | 结构/维度 | 本项目输入合同 |
|---|---|---|---|
| 中文 RoBERTa | `hfl/chinese-roberta-wwm-ext` | 双向 encoder，768 维 | 最大 512 token；保存 prompt/title/body/full mean/max、CLS 和 prompt token |
| BGE-M3 | `BAAI/bge-m3` | 双向多语言 embedding encoder，1024 维 | 最大 1000 token；输出与 RoBERTa 分目录、同类型 pooled/token 产物 |
| Qwen3-Embedding-8B | 本地/Ollama `qwen3-embedding:8b` | 因果模型，4096 维 | prompt 放在正文后；正文用 `article_mean`，目标 span 排除句号 |

### 3.1 Embedding 用时和容量估算

六个中性 Prompt 的正式 CPU 任务按 256 个 shard、最大并行 256、每 task 8 CPU/32 GB、
24 小时时限运行；每个 task 只加载一次模型并连续编码六个 Prompt。下表的分片统计只使用
Slurm `ElapsedRaw` 已完成任务；“整组墙钟”从首个运行分片到最后完成分片，不含排队时间。

| 数据/模型 | 完成分片 | 单分片中位数 / P90 / 最大值 | 已观察整组墙钟 | 解释 |
|---|---:|---:|---:|---|
| 新浪 RoBERTa | 256/256 | 46分50秒 / 3时04分36秒 / 3时32分12秒 | **4时28分17秒** | 全量 796,553 行完成 |
| 巨潮 RoBERTa | 255/256 | 1时11分10秒 / 2时01分47秒 / 4时33分41秒 | 9时16分57秒 | 基础 array 缺 shard-225，需以补跑产物验收 |
| 新浪 BGE-M3 | 189/256 | 5时15分51秒 / 9时07分35秒 / 15时19分15秒 | 15时49分45秒 | 其余 67 个取消，墙钟不是全量完成时间 |
| 巨潮 BGE-M3 | 78/256 | 4时36分44秒 / 4时54分07秒 / 6时42分05秒 | 8时28分01秒 | 其余 178 个取消，墙钟不是全量完成时间 |

在健康 CPU 节点且 256 个槽位能同时启动的条件下，应按 P90 而不是最快节点估算，另加
至少 20% I/O/校验余量；BGE-M3 单个全量 array 的条件预算约为新浪 10--16 小时、巨潮
5--7 小时，**排队等待另计**。通用公式为
`ceil(shards/effective_concurrency) × representative_shard_elapsed`。

Qwen 的可复核参考不是六 Prompt 全量 token 任务：历史 job 3651805 在一张 RTX 4090 上
用 5时11分55秒生成 57,741 行、4,096 维 pooled embedding，约 3.085 行/秒。相同吞吐下
单次 pooled pass 线性外推为新浪约 71.7 小时、巨潮约 81.4 小时；token 输出、不同长度、
缓存和多 GPU 会改变速度，不能把这个数字当成六 Prompt 的承诺。详细证据见
[Embedding 用时审计](references/embedding_runtime_audit_20260826.md)。

下面只比较已有结果中严格同口径的两个模型：新浪共同面板、2018--2026 严格 6+2+1
滚动、`next_day_return`、`masked_short`、各 Prompt 自身目标 span、无聚类线性 Ridge。

| Prompt 目标 span | RoBERTa RankIC | BGE-M3 RankIC |
|---|---:|---:|
| 盈利 | 0.05430 | **0.03812** |
| 收益 | 0.05394 | 0.03556 |
| 超额收益 | **0.05644** | 0.03720 |
| 亏损 | 0.05479 | 0.03431 |

四个 Prompt 在两个模型中均为正；RoBERTa 内部以“超额收益”最高，BGE-M3 内部以
“盈利”最高。因此结果支持方向词 token 含有收益排序信息，但不支持“收益”一词在所有
模型中普遍最优。三模型维度和结构不同，不能直接比较原始坐标；跨模型只比较同新闻、
同标签、同 PCA 算法下的标准化统计和样本外因子。完整口径和新 Prompt 对照见
[综合研究报告](reports/comprehensive_research_report/report.md)。

## 4. Benchmark 与扩展实验

### 4.1 论文 benchmark

原始论文的可复现骨架是“冻结文本表示 + 简单监督头 + 严格滚动 + 组合检验”。仓库保留
TF-IDF/词典/Word2Vec/pooled Transformer 等传统或整文基线；当前 prompt-token 主线的
统一 benchmark 是：

```text
同一新闻交集 -> 训练期 StandardScaler -> 训练期 randomized PCA32
-> Ridge(alpha=100) -> 股票日聚合 -> 测试期 RankIC/组合
```

新浪使用 6 年训练 + 2 年验证 + 1 年测试，测试年 2018--2026；巨潮公平比较按用户
确定的 3 年历史 + 1 年测试。所有 PCA、scaler、聚类器和监督模型只在历史训练数据拟合。
不做原始 768/1024/4096 维回归，也不根据测试年切换参数。

### 4.2 Mask 与 prompt 扩展

`masked_short` 只 mask 公司身份、股票代码、日期和时间，不 mask 盈利、收益、亏损、
估值、风险等目标词。mask 的作用是控制身份/时间泄漏，不假设它一定提高收益预测；已有
16 对配置中只做多 Sharpe 平均增加 0.069，但 `simple_states` IC 平均下降 0.00480。

方向 prompt 为 `分析股票盈利/收益/超额收益/亏损`。中性任务 prompt 为
`分析股票估值/确定性/波动率/冲击/流动性/风险`。目标 span 只取目标词对应 token 的
均值，不含“分析股票”、句号、separator 或 special token。正文必须用 `body_mean`；
Qwen 对应用 `article_mean`，禁止用 `full_mean` 冒充正文。

`prompt_token_embeddings` 的 `N × P × D` 表示 `N` 条新闻、每条保存 `P` 个 Prompt
token、每个 token 是 `D` 维 hidden state；不是三个数相乘后的标量。目标 span 或
`prompt_mean` 沿 `P` 维取均值后才得到供 PCA 使用的 `N × D` 矩阵。

扩展实验还把语义 prompt 对齐到对应标签：估值使用 PE/PB/PS/EV-EBITDA 的滞后
log 相对偏离；波动率使用下一日实现波动率水平、log 增量和 5 日相对 20 日跳升；
流动性使用下一日价差水平和 log 变化。它们回答的是不同预测任务，不能都解释为收益率。

收益回归目前有两个不同口径：四方向 masked target span 对次日收益采用 2018--2026
严格 6+2+1，RoBERTa/BGE-M3 四词平均 RankIC 为 0.05486/0.03630；18 条高中低 Prompt
构成的六个 direction 轴对未来三日收益只有 2026 单折，PCA+Ridge token RankIC 为
确定性 0.05623、期限收益 0.04311、波动率 0.04298、流动性 0.03324、估值 0.02241、
冲击 0.01501。六条单独中性 Prompt 目前没有同一收益标签回归，不能把 direction 轴结果
写成单独 `分析股票波动率` 的结果；两个已完成口径也不能按绝对 RankIC 直接排名。

## 5. PCA 与聚类方法

聚类不是重复做一份无监督可视化，而是在同一训练窗 PCA 表示上检验非线性分层是否
给收益预测带来增量。

| 方法 | 原理 | 本项目实现 | 如何判断有效 |
|---|---|---|---|
| PCA32 + Ridge | PCA 提取训练期最大方差方向，Ridge 对连续收益做线性收缩 | randomized PCA32，seed 42，Ridge alpha 100 | 所有聚类方法必须与此配对 |
| 硬 KMeans | 将样本分到最近质心，得到离散语义状态 | MiniBatchKMeans k=6；簇 one-hot/距离与 PCA 特征进入 Ridge | OOS RankIC/组合增量；当前 16 对平均 -0.00026 |
| 软 KMeans | 把到各质心的距离变成连续 membership，避免边界跳变 | 距离 softmax，temperature 0.5；训练期簇收益向全局均值 shrinkage 500 | 只用历史收益估计簇均值，测试期冻结 |
| UMAP + HDBSCAN | UMAP 保留局部邻域；HDBSCAN 发现不规则密度簇并允许噪声点 | PCA -> UMAP8（cosine, neighbors 30, min_dist 0.1）-> HDBSCAN（50/10）-> Ridge | 仅作非线性复核；不能用 silhouette 选择收益模型 |
| GMM | 假设 PCA 空间由多个高斯成分混合，以后验概率表示软状态 | Qwen 聚类分析中使用 k=3/5/8，输出 posterior/簇年度收益 | 描述和稳健性复核，不是当前主模型选择器 |

ARI、silhouette、簇规模和簇收益排序只描述结构。模型参数只能由验证期 RankIC 或冻结
配置确定，不能用全样本聚类图选择最终收益模型。

## 6. 当前产物完整性

| 数据集/模型 | 中性六 prompt 分片 | 状态 |
|---|---:|---|
| 新浪 RoBERTa | 256/256 | 完整 |
| 巨潮 RoBERTa | 255/256 | 缺 shard-225 |
| 新浪 BGE-M3 | 189/256 | 剩余任务已取消，保留现有完成分片 |
| 巨潮 BGE-M3 | 78/256 | 剩余任务已取消，保留现有完成分片 |

三模型四方向 prompt 公平比较使用预计最小交集：新浪 35,576 条、巨潮 254,544 条。
截至快照 build 任务仍因 Priority 等待，后续 geometry/rolling/fusion 为依赖等待，
因此这些公平比较结果尚未生成，不能把计划写成结论。

重新审计数据、embedding 和 Slurm 状态：

```bash
python scripts/audit_research_handoff.py --include-slurm
```

## 7. 评价与验收

主指标是日均 RankIC、RankIC IR、逐年 RankIC 和正 RankIC 年份数。辅助报告月度正 IC
比例、日期区块 bootstrap 95% 区间、OOS R2、Top20 同新闻池超额、多空/只做多收益、
5bp 单边成本后的 CAGR、Sharpe、最大回撤、因子相关和 Top20 选股重合率。

至少满足以下约束才可写成稳定增量：测试期完全样本外；同新闻池配对；至少 6/9 年方向
一致（巨潮短协议按可用测试年另报）；bootstrap 区间不跨 0；成本或风险指标至少一项
改善；所有结论能回指 CSV/manifest，而不是只存在于文字报告。

## 8. 仓库与运行入口

```text
configs/      路径、样本、模型和新浪年度配额
data/         小型样本、目录占位和股票池；大型授权数据不入 Git
docs/         交接、运行手册和状态快照
references/   论文清单、论文映射、数据/模型合同
src/          可导入、可测试的数据、模型、评价和组合逻辑
scripts/      采集、清洗、embedding、滚动实验、审计和 Slurm 入口
reports/      可追溯结果、表格和综合报告源码
tests/        单元测试、时间泄漏和采集器测试
```

服务器大型数据统一从以下共享入口访问：

```text
/data/alpha_team2/shares/llm_return/
```

其中 `datasets/repository_data`、`datasets/sina_cninfo_full_2010_2026`、
`datasets/news_content_quality_20260812` 分别对应仓库数据、全量新浪/巨潮主目录和历史清洗
产物；`external/intraday_valuation_260825` 对应高频波动率、价差和估值数据。完整映射和权限
说明见共享目录的 `README.md`、`SOURCE_PATHS.tsv` 和 `SHARE_AUDIT.md`。大型文件通过同一
Lustre 文件系统的共享链接提供，不重复进入 Git 或复制数 TB 数据块。

常用入口：

```bash
# 环境与测试
python -m pip install -e '.[dev]'
pytest -q

# 只读状态审计
python scripts/audit_research_handoff.py --include-slurm

# 用股票日预测执行成本回测
python scripts/run_portfolio_strategy.py PREDICTIONS.parquet \
  --output-dir reports/strategy/example --cost-model paper --gammas 1.0

# 重建综合 PDF
python reports/comprehensive_research_report/build_report.py
```

## 9. 数据与合规规范

- 原始新闻/PDF、模型权重、embedding、任务快照和授权行情不进入 Git。
- 每次数据传输保留来源、采集时间、参数、行数、稳定 ID、SHA256 和 manifest。
- 不绕过 robots、登录、验证码、访问控制或网站频率限制。
- 新闻发布时间必须映射到当时可交易的下一时点；收盘后和非交易日新闻不能前视。
- 标准化、PCA、聚类、标签选择和监督模型都只能用训练/验证期允许的信息。
- Qwen 为因果模型且 prompt 后置；RoBERTa/BGE-M3 为双向模型，正文表示的含义不同。
- 旧新浪 75,894 行面板、全量新浪 796,553 行面板和巨潮 903,665 行面板禁止共用 row_index。

## 10. 文档索引

- [完整交接和当前任务](docs/HANDOFF_2026-08-26.md)
- [采集、运行与恢复手册](docs/RUNBOOK.md)
- [综合研究报告](reports/comprehensive_research_report/report.md)
- [数据源和模型说明](references/data_sources_and_models.md)
- [论文到实验映射](references/paper_to_experiment_map.md)
- [完整复现计划](references/full_replication_plan.md)
- [报告索引](reports/README.md)

仓库当前只保存可公开、可复现的代码和小型事实文件。任何人接手前，应先运行只读审计，
确认本机/Lustre 的大文件状态与本快照一致，再继续计算。
