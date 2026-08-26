# 数据来源、模型与表示合同

本文档记录已经使用的数据和模型，不再是前期选型建议。完整采集参数、实测结果和路径见
`docs/HANDOFF_2026-08-26.md`。

## 1. 数据源

| 来源 | 内容 | 当前研究面板 | 主要限制 |
|---|---|---:|---|
| 新浪财经 | 历史财经新闻标题/正文 | 796,553 条，2010--2026 | 联网采集只能本地运行；历史年份覆盖不均 |
| 巨潮资讯 | 官方公司公告详情和 PDF | 903,665 条、816 股票，2010--2026 | 公告长文本和股票池与新浪不同 |
| 东方财富 | 要闻、个股资讯、公告、研报、股吧 | 有界原型 | 公开可见浏览器页、低频批次 |
| 雪球 | 公开讨论、资讯和公告 | 有界原型 | 匿名或显式授权 browser state |
| 日频行情 | OHLC、复权、停牌/涨跌停、行业和市值 | 与新闻/公告按点时规则对齐 | 收盘后新闻只能用于下一可交易时点 |
| 高频/估值 | 1 分钟波动率/价差；PE/PB/PS/EV-EBITDA | 仅使用实际日期交集 | `/data/alpha_team2/shares/260825/` 不入 Git |

### 新浪

浏览器脚本负责发现和验证少量动态历史 seed；普通 HTTP 脚本负责有界并发、原始 HTML、
JSONL、manifest、state 和断点恢复。推荐组合为“浏览器发现 + HTTP 扩展”，而不是全浏览器
抓正文。新浪所有联网命令只能在本地机器执行，完成后连同 SHA256 复制到 Lustre。

### 巨潮

`collect_cninfo_announcements.py` 使用公开公司披露页的可见日期筛选、分页、详情和 PDF
链接。2010--2017 的 6,528 个股票年审计全部完成，合并后最终面板 903,665 条。
采集、PDF 文本抽取、去重清洗、分类面板和收益对齐是四个独立阶段，不能把最终面板条数
写成原始 PDF 下载数。

### 东方财富和雪球

共用 `collect_browser_visible.py`，只采集公开可见内容，遇验证码/访问异常立即停止。
当前 batch 默认每批 3 股票，页面间隔至少 3 秒，批间 60 秒。这两类数据主要用于投资者
情绪和注意力控制，不是新浪/巨潮主文本库的替代品。

## 2. 最低字段与稳定键

文本记录至少包含：

```text
row_index, document_id, stock_id, published_at, source,
headline/title, body/text, body_sha256, collected_at
```

最终建模面板还应包含：

```text
trading_date, next_day_return, event_return_3d, o2o_return,
industry, market_cap, is_suspended, is_limit_up, is_limit_down
```

每个数据集独立生成 `row_index`。跨数据集或重建版本只按稳定 `document_id`、股票、日期和
正文哈希对齐，不能假设行号可复用。

## 3. 时间对齐

- 收盘前新闻用于下一交易日预测；收盘后新闻从下一个可交易时点开始；
- 周末/节假日新闻映射到下一交易日；
- 转载和更新按正文哈希、URL、标题和发布时间去重，保留最早可用版本；
- 任何 scaler、PCA、词表、聚类器、簇收益和监督模型都只使用训练期；
- 估值、波动率和价差标签使用下一日或明确的未来窗口，输入特征必须先滞后。

## 4. 模型

| 模型 | 标识/来源 | 结构 | hidden | 最大输入 | 正文表示 |
|---|---|---|---:|---:|---|
| 中文 RoBERTa | `hfl/chinese-roberta-wwm-ext` | 双向 encoder | 768 | 512 | `body_mean` |
| BGE-M3 | `BAAI/bge-m3` | 双向多语言 embedding encoder | 1024 | 1000 | `body_mean` |
| Qwen3-Embedding-8B | Ollama `qwen3-embedding:8b` | 因果模型 | 4096 | 既有 runner 合同 | `article_mean` |

RoBERTa/BGE-M3 输出：

```text
prompt_mean, title_mean, body_mean, title_body_mean, full_mean, cls,
title_max, body_max, title_body_max, full_max, prompt_token_embeddings
```

每个模型还保存 tokenizer/model 标识、输入 IDs、prompt tokens、行 metadata、prompt spec、
manifest 和 `COMPLETED`。不同模型维度、tokenizer 和目录不可混用。

Qwen 的 prompt 位于正文后，因果注意力下 `article_mean` 不受后置 prompt 反向影响；
RoBERTa/BGE-M3 为双向模型，prompt 会参与正文编码。这一差异必须出现在报告中。

## 5. Prompt 与 mask

方向 prompt：

```text
分析股票盈利
分析股票收益
分析股票超额收益
分析股票亏损
```

中性 prompt：

```text
分析股票估值
分析股票确定性
分析股票波动率
分析股票冲击
分析股票流动性
分析股票风险
```

只使用 `masked_short`：mask 公司身份、代码、日期和时间，不 mask 目标语义。目标 token
只取目标 span，不含前缀、句号和 special token。不同 prompt 不添加无意义 padding；
实际 span 以 tokenizer input IDs/offset 为准。

## 6. 实测性能边界

| 结果 | RankIC | 说明 |
|---|---:|---|
| 新浪 RoBERTa+BGE masked 收益 span event-3 | 0.08945 | PCA64 分别降维后拼接，9/9 年为正 |
| 新浪 RoBERTa masked 收益 prompt mean next-day | 0.06044 | 9/9 年为正 |
| 巨潮 RoBERTa pooled | 0.01764 | 9/9 年为正 |
| 巨潮 BGE-M3 pooled | 0.01415 | 9/9 年为正 |
| Qwen 新浪正文 / 收益 token | 0.05680 / 0.05360 | token 尚未超过正文 |
| Qwen 巨潮 residual O2O | 0.02732 | 3+1+1，5/5 年为正 |

这些结果来自不同任务和历史配置，只证明实现可用，不能直接做模型排行榜。三模型公平
比较必须使用同新闻、同 prompt、同标签和各自训练期 PCA32。

## 7. 数据与模型合规

- 新闻正文、公告 PDF、模型权重、embedding 和授权行情不进入 Git；
- API key 只通过环境变量，禁止进入代码、YAML、命令行历史或日志；
- 未获明确授权时不向外部 embedding API 上传新闻正文；
- 固定模型 ID、revision/tokenizer hash、最大长度、batch、文本哈希和输出维度；
- 模型“性能”只报告本项目严格样本外指标，不用发布时间、参数规模或公开榜单替代。
