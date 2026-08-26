# 运行手册

## 1. 环境

```bash
cd /mnt/lustre3/home/gaozh/llm_return
python -m venv .venv
.venv/bin/pip install -r requirements-research.txt
```

2026-08-26 实际环境的关键版本为：Python 环境中的 NumPy 2.0.2、Pandas
2.3.3、SciPy 1.13.1、scikit-learn 1.6.1、PyArrow 21.0.0、Torch 2.8.0、
Transformers 4.57.6、UMAP 0.5.12、HDBSCAN 0.8.42、XGBoost 2.1.4、
LightGBM 4.6.0、CatBoost 1.2.10。

## 2. 数据采集入口

### 新浪：只能本地运行

新浪所有联网发现和抓取都在本地 Windows/工作站运行，禁止提交 Slurm。推荐先用浏览器
发现少量种子，再用普通 HTTP 扩正文。

```powershell
$env:PYTHONPATH = "scripts"

# 浏览器检查/筛选历史种子
python scripts/select_sina_historical_seeds_browser.py `
  --root-file data/interim/sina_roots.txt `
  --years 2010 2011 2012 2013 2014 2015 2016 2017 `
  --per-year 20 --workers 8 --pause-seconds 3 `
  --output data/interim/sina_seed_selection.json `
  --catalog-output data/interim/sina_seed_catalog.csv

# 普通 HTTP depth-3 -> 去重 -> depth-5
python scripts/run_sina_auto_pipeline.py `
  --root-file data/interim/sina_roots.txt `
  --stock-catalog data/stock_universe_csi500_current.csv `
  --output-dir data/interim/sina_auto_v1 `
  --workers 20 --pause-seconds 1 --timeout 30
```

本地完成后将 raw HTML、records JSONL、state、manifest 一起复制到 Lustre，并生成
SHA256。不要只复制最终 JSON；否则无法审计和恢复。

### 巨潮、东方财富和雪球

```bash
# 巨潮：公开公司页/详情/PDF；先用小日期范围和小股票池
python scripts/collect_cninfo_announcements.py --help

# 东方财富/雪球：公开可见浏览器页；默认页面间隔 3 秒
python scripts/collect_browser_visible.py --help
python scripts/run_100_stock_collection.py --help
python scripts/run_social_collection_batches.py --help
```

需要 browser storage state 时只能使用用户显式提供且已授权的文件。检测到验证码、访问
异常或请求过频就停止，不增加代理、隐藏 API 或绕过逻辑。

## 3. 只读状态审计

```bash
.venv/bin/python scripts/audit_research_handoff.py --include-slurm
```

该命令检查：新浪/巨潮面板行数、全量中性输入 manifest、四组 RoBERTa/BGE-M3
`COMPLETED` 数量、缺失 shard、主报告哈希和当前 Slurm 队列。它不修改数据。

## 4. 主线代码

### 中性六 Prompt embedding

- 配置：`configs/prompts/neutral_masked_short_v1.json`
- 输入：`scripts/build_neutral_masked_inputs.py`
- 单 shard runner：`scripts/run_neutral_masked_embeddings.py`
- Slurm：`scripts/slurm_neutral_masked_embeddings.sbatch`
- 提交：`scripts/submit_neutral_masked_embeddings.sh`

Prompt 严格为：分析股票估值、确定性、波动率、冲击、流动性、风险。每个 shard
只加载一次模型并连续计算六个 prompt，输出 prompt/body/full 等 pooled embedding
和 prompt token embedding。

当前 BGE-M3 数组已主动取消。不要直接再次运行完整提交脚本；恢复前先由状态审计
生成缺失 shard 清单，并确认 Fairshare 与并发上限。已完成输出是原子落盘的，未完成
task 没有可恢复检查点。

### 三模型四 Prompt PCA32 公平比较

- 构建交集：`scripts/build_three_model_four_prompt_fair.py`
- 几何：`scripts/analyze_three_model_prompt_geometry.py`
- 单 fold：`scripts/run_three_model_prompt_fair_fold.py`
- 融合：`scripts/run_three_model_prompt_fusion.py`
- 汇总：`scripts/summarize_three_model_prompt_fair.py`
- 提交：`scripts/submit_three_model_four_prompt_fair.sh`

```bash
bash scripts/submit_three_model_four_prompt_fair.sh
```

提交必须经过 `scripts/task_tracker.py`。当前对应根任务记录为
`20260826T025226Z-three-model-four-prompt-fair-pca-v2-4035915`；已提交 Job
`5043643--5043652`，但快照时仍因 Fairshare 等待。不要重复提交相同输出目录。

固定参数：PCA32、Ridge alpha 100、KMeans k 6、seeds 17/29/42/71/113、
soft shrinkage 500、temperature 0.5、UMAP8、HDBSCAN min cluster size 50。
不运行原始 embedding 回归，也不按最终测试结果改参数。

### 巨潮预测级树模型

- runner：`scripts/run_cninfo_oos_tree_stacking.py`
- Slurm：`scripts/slurm_cninfo_oos_tree_stacking.sbatch`
- 提交：`scripts/submit_cninfo_oos_tree_stacking.sh`
- 报告更新：`scripts/update_report_cninfo_tree_stacking.py`

树模型使用 72 个已经样本外生成的一级预测因子，不拼接 768/1024 维原始 embedding。
训练 2018--2023，验证 2024--2025，封存测试 2026。快照时 Job 5036181/5036182
仍在排队，尚无巨潮树模型结果。

## 5. 本地验证

```bash
.venv/bin/python -m compileall -q src scripts
.venv/bin/pytest -q
.venv/bin/python scripts/audit_research_handoff.py
```

对主线做快速聚焦测试：

```bash
.venv/bin/pytest -q \
  tests/test_audit_research_handoff.py \
  tests/test_three_model_four_prompt_fair.py \
  tests/test_bge_embedding_recovery.py \
  tests/test_task_tracker.py
```

## 6. 输出验收

一个 embedding shard 只有同时存在根目录 `COMPLETED`、六个 prompt 子目录、
manifest/preflight、metadata、pooled arrays 和 prompt token arrays 才算完成。仅有目录或
临时文件不能计数。

一个 rolling fold 只有在 `COMPLETED`、`manifest.json`、`fixed_parameters.json`、
`metrics.csv` 和 test prediction parquet 同时存在时才算完成。PCA/scaler/clusterer 必须
只在该 fold 的历史窗口拟合。

## 7. Git 与大型产物

Git 只跟踪源码、配置、测试和文档。以下内容不可普通提交：

- `models/` 中约 451MB 的 FinBERT 权重；
- `task_records/` 中约 7.4GB 的任务代码快照；
- PDF、授权数据、embedding、预测 parquet、logs 和 CatBoost 运行目录。

如确需共享二进制，应先确认授权，再使用受控对象存储或 Git LFS，不能取消
`.gitignore` 后直接推送。

## 8. GitHub 与 GitLab

当前 GitHub `origin` 为 `git@github.com:2400010793/llm_return.git`。提交前先拉取远端、
运行测试并审计大文件；不要强推覆盖他人提交。

快照时没有可用的 GitLab remote/SSH key。获得 GitLab 私有仓库 URL 后执行：

```bash
git remote add gitlab <GITLAB_REPOSITORY_URL>
git push -u gitlab main
```

推送前必须再次运行测试、`git status` 和大文件审计，并确认 PDF/模型/数据没有进入
暂存区。
