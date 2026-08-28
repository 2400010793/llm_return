#!/usr/bin/env bash
set -euo pipefail

cd /home/team/llm_return

# 等待当前东方财富新闻/公告和研报任务自然结束，不中断任何已有进程。
while pgrep -f 'run_100_stock_collection.py.*stock1000_eastmoney_(ann_news|research)' >/dev/null; do
    printf '[%s] waiting for existing Eastmoney jobs\n' "$(date -Is)"
    sleep 60
done

printf '[%s] existing Eastmoney jobs finished; starting 100-detail supplement\n' "$(date -Is)"
exec /home/team/llm_return/.venv/bin/python scripts/run_100_stock_collection.py \
    --stocks data/stock_universe_paper_1000.csv \
    --expected-stocks 1000 \
    --output-dir data/interim/stock1000_eastmoney_100 \
    --manifest data/interim/collector_manifest_stock1000_eastmoney_ann_news.json \
    --batch-size 3 \
    --start-batch 1 \
    --max-batches 334 \
    --batch-pause 60 \
    --page-pause 6 \
    --stock-detail-limit 100 \
    --research-detail-limit 0 \
    --forum-detail-limit 0 \
    --stock-links-only