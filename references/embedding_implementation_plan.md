# 文本表示与 Embedding 实施计划

## 目标

在不引入前视偏差的前提下，将中国财经新闻转换为可用于收益预测的数值表示，并严格比较：

1. 词袋/TF-IDF 表示；
2. 词典情绪表示；
3. Word2Vec 表示；
4. 中文 BERT/RoBERTa 表示；
5. BGE-M3 句向量表示；
6. 可选的 API embedding。

第一版只完成“文本表示”，不直接训练复杂的生成式大模型。收益预测模型统一先使用 Ridge/Elastic Net，以便把差异归因于文本表示本身。

## 一、总体流程

```text
原始新闻
  -> 去重、清洗、时间统一
  -> 新闻与股票实体匹配
  -> 按训练/验证/测试时间切分
  -> 只在训练集拟合词表、TF-IDF、词典筛选和降维
  -> 生成文本表示
  -> 缓存为 Parquet/NPZ
  -> 滚动窗口收益预测
  -> 分组组合和样本外评价
```

禁止在全样本上先拟合 TF-IDF、词频筛选、Word2Vec 或降维后再切分时间；这会把未来文本分布泄露给训练期。

## 二、依赖安装顺序

### 第 1 批：CPU 可运行的基线

- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `pyarrow`
- `pyyaml`
- `tqdm`
- `joblib`
- `jieba`
- `statsmodels`

先用这批依赖完成数据清洗、TF-IDF 和 Ridge。不要在数据还未通过审计前下载大模型。

### 第 2 批：本地 Transformer

- `torch`
- `transformers`
- `sentence-transformers`
- `safetensors`

安装前确认 CUDA、显存和磁盘空间。若没有 GPU，仍可用小样本 CPU 测试，但不建议直接处理全量新闻。

### 第 3 批：可选扩展

- `gensim`：Word2Vec/Doc2Vec；
- `openai` 或其他服务商 SDK：仅在数据授权允许外发文本且确认 API 成本后使用。

## 三、模型分层

### A. 词袋基线

#### A1. CountVectorizer

输出词频矩阵，作为最简单对照。

#### A2. TF-IDF + Ridge

第一版主基线。建议参数网格：

- `ngram_range`: `(1, 1)`、`(1, 2)`；
- `min_df`: 5、10、20；
- `max_df`: 0.95、1.0；
- `sublinear_tf`: `True`；
- `max_features`: 根据内存设为 50,000—200,000；
- Ridge `alpha`: 1、10、100。

中文文本可以先按字符 n-gram 做稳健基线，再比较分词后的词 n-gram。所有参数只能用训练窗口选择。

缓存内容：

- vocabulary；
- idf；
- tokenizer/分词版本；
- 训练窗口；
- 稀疏矩阵文件；
- 参数配置。

### B. 词典情绪

建立统一的中文金融情绪接口，至少输出：

- 正面词数量；
- 负面词数量；
- 情绪净值；
- 情绪词占比；
- 否定词修正后的情绪值；
- 标题情绪和正文情绪分别计算。

词典来源、版本和授权必须写入 `references/`。不要把中文词典和英文 Loughran-McDonald 词典直接混用；跨语言比较应单独报告。

### C. Word2Vec

只作为传统词向量对照，不作为第一批必做模型。两种方案：

1. 使用预训练中文 Word2Vec；
2. 只在训练窗口内训练 Word2Vec，再将新闻词向量做平均/TF-IDF 加权平均。

第二种更符合严格样本外要求，但计算成本更高。不能使用利用全时期文本训练的词向量来声称严格样本外结果。

### D. 中文 BERT/RoBERTa

优先顺序：

1. `hfl/chinese-roberta-wwm-ext`；
2. `hfl/chinese-bert-wwm-ext`。

第一版只做冻结模型推理，不进行微调：

- tokenizer 截断长度先设 256；
- 标题与正文拼接，并保留长度统计；
- 使用 `[CLS]` 向量或 mean pooling；
- 长文本采用分块编码后 mean pooling；
- 每篇新闻保存一个固定维度向量；
- 批量推理并保存缓存，避免重复下载和计算。

BERT/RoBERTa 的模型参数可以使用公开预训练模型，但不能在全样本新闻上继续预训练，除非训练过程严格按时间滚动并单独记录。

### E. BGE-M3

`BAAI/bge-m3` 作为第二阶段增强模型：

- 适合中文和多语言文本；
- 使用 sentence-transformers 或官方 Transformers 接口；
- 对超过最大长度的正文分块；
- 对块向量做平均或长度加权平均；
- 记录模型版本、维度、最大长度和硬件信息。

先在 1,000—10,000 篇新闻上测试速度、显存和向量质量，再决定是否跑全量。

## 四、建议的代码文件

第一批实现：

- `src/text/preprocess_zh.py`：清洗、分句、去重、中文规范化；
- `src/text/bow_features.py`：CountVectorizer、TF-IDF 和稀疏矩阵缓存；
- `src/text/sentiment_lexicon.py`：词典情绪特征；
- `src/text/embeddings.py`：Transformer embedding 的统一接口；
- `src/text/cache.py`：向量、词表和元数据缓存；
- `src/text/quality_checks.py`：文本泄漏、空文本、重复新闻和长度检查；
- `notebooks/03_text_baselines.ipynb`：词袋/词典基线；
- `notebooks/04_llm_embeddings.ipynb`：BERT/RoBERTa/BGE-M3 推理。

每个模型接口都应返回：

```text
article_id
stock_id
published_at
model_name
model_version
embedding_path 或 sparse_matrix_path
embedding_dim
text_hash
processing_timestamp
```

## 五、分阶段实验

### 实验 0：文本质量审计

- 空标题/正文比例；
- 重复新闻比例；
- 每只股票新闻数量；
- 每日新闻数量；
- 新闻长度分布；
- 中文、英文和混合文本比例；
- 收盘前后新闻比例。

### 实验 1：词袋最小版本

- TF-IDF + Ridge；
- 预测下一交易日收益；
- 3 年训练窗口，滚动预测；
- 输出 MSE、方向准确率、预测值分组收益。

### 实验 2：中文 BERT/RoBERTa

在完全相同的股票样本、标签、训练窗口和组合形成规则下，与 TF-IDF 对比。此时只改变文本表示，不改变预测模型。

### 实验 3：BGE-M3

在前两个模型流程稳定后加入，优先使用 1 年小样本。记录运行时间、显存、缓存大小和预测表现。

### 实验 4：联合模型和消融分析

比较：

- 仅 LLM embedding；
- 仅 RavenPack/词典情绪；
- LLM + 词典情绪；
- LLM + 股价特征；
- LLM + 百度指数/股吧情绪。

## 六、资源和缓存策略

- 原始新闻只读保存，不覆盖；
- 清洗后的新闻保存为 Parquet；
- 稀疏 TF-IDF 保存为 `.npz`，词表和 ID 保存为 JSON/Parquet；
- 密集 embedding 按日期或月份分片保存为 `.npy`/Parquet；
- 每个缓存必须带 `model_name`、`model_version`、`text_hash` 和参数文件；
- 不要把模型文件、原始新闻和大规模 embedding 提交到 Git；
- 先保存 1 个月小样本缓存，再扩展全样本。

## 七、验收标准

只有满足以下条件，才进入下一阶段：

1. 新闻与股票的时间对齐通过人工抽样；
2. 词袋拟合范围没有使用未来文本；
3. 同一 `text_hash` 能复用缓存；
4. Transformer 推理可重复，向量维度固定；
5. 随机抽取的新闻不会出现空向量或错位 ID；
6. 训练、验证、测试结果可由配置文件复现；
7. 预测组合回测明确处理停牌、涨跌停、T+1、卖空和交易成本。

## 八、当前决策

当前不安装或下载所有大模型。正确顺序是：

1. 确认新闻数据样本和字段；
2. 完成数据审计；
3. 安装第 1 批依赖并跑通 TF-IDF；
4. 确认 GPU/CPU 资源；
5. 下载 `chinese-roberta-wwm-ext`；
6. 完成小样本 embedding 和缓存；
7. 再决定是否下载 `chinese-bert-wwm-ext` 和 `bge-m3`。

## 九、已有 Ollama 模型的使用方式

当前机器已发现 Ollama 二进制和约 19 GB 的本地模型权重；日志显示模型为 Qwen2.5 32B Instruct 的 Q4_K_M 量化版本，并且曾在约 47 GB 显存的 NVIDIA GPU 上运行。Ollama 服务当前未运行，因此不能直接调用 `ollama list` 或 HTTP API。

该 Qwen 模型是对话/文本生成模型，不应直接当作第一版 embedding 模型。它可以作为辅助实验：

- 新闻情绪分类；
- 事件类型抽取；
- 新闻摘要或结构化字段抽取。

主实验仍建议使用 TF-IDF 和专用句向量模型，保证速度、维度稳定和结果可重复。Ollama 模型若用于辅助信号，必须固定模型标签、提示词、温度为 0、输出格式和服务版本，并缓存每条新闻的结果。
