"""Evaluate independent prompt predictions and equal-weight late fusion.

Fusion is prediction-level only: each prompt is trained independently and the
fusion uses the common stock-day intersection. Costs are actual weight changes
(buy 5bp, sell 10bp) with final liquidation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROMPTS=("profit","excess_return","return","loss")
WORD_SPANS={
    "profit": "profit_span",
    "excess_return": "excess_span",
    "return": "plain_return_span",
    "loss": "loss_span",
}

def rank_series(s): return s.rank(method="average", pct=True)

def portfolio(frame, score, fraction=0.20, strategy="long", buy=0.0005, sell=0.001):
    frame=frame.copy(); frame["score"]=score; previous={}; daily=[]
    for date,g in frame.groupby("entry_date",sort=True):
        g=g.sort_values(["score","stock_id"],kind="mergesort"); n=min(max(1,int(np.floor(len(g)*fraction))),len(g)//2)
        if n<1: continue
        if strategy=="long": weights={str(x):1/n for x in g.tail(n).stock_id}
        else:
            hi={str(x):0.5/n for x in g.tail(n).stock_id}; lo={str(x):-0.5/n for x in g.head(n).stock_id}; weights={**hi,**lo}
        names=set(previous)|set(weights); bought=sum(max(weights.get(x,0)-previous.get(x,0),0) for x in names); sold=sum(max(previous.get(x,0)-weights.get(x,0),0) for x in names)
        outcomes=dict(zip(g.stock_id.astype(str),g.actual_return.astype(float))); gross=sum(w*outcomes.get(x,0) for x,w in weights.items()); cost=buy*bought+sell*sold
        benchmark=float(g.actual_return.mean())
        daily.append((date,gross,gross-cost,cost,benchmark)); previous=weights
    if previous:
        bought=sum(max(-w,0) for w in previous.values()); sold=sum(max(w,0) for w in previous.values()); daily.append(("liquidation",0,-(buy*bought+sell*sold),buy*bought+sell*sold,0))
    d=pd.DataFrame(daily,columns=["date","gross","net","cost","benchmark"])
    return dict(gross=d.gross.sum(),net=d.net.sum(),cost=d.cost.sum(),gross_excess=(d.gross-d.benchmark).sum(),net_excess=(d.net-d.benchmark).sum(),days=max(0,len(d)-1),positive_days=float((d.iloc[:-1].net>0).mean()) if len(d)>1 else np.nan)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--model',choices=('roberta','bge_m3'),default='roberta'); p.add_argument('--variant',choices=('short','masked_short'),default='masked_short'); p.add_argument('--representation',default='prompt_mean'); p.add_argument('--word-spans',action='store_true',help='Use each prompt direction word: 盈利/超额/收益/亏损'); p.add_argument('--cluster',action='store_true',help='Read PCA+MiniBatchKMeans Ridge predictions'); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    representation_label='word_span' if args.word_spans else args.representation
    tables=[]
    for prompt in PROMPTS:
        representation=WORD_SPANS[prompt] if args.word_spans else args.representation
        if args.cluster:
            path=args.root/'results_cluster'/'sina'/f'{prompt}_{args.model}_{args.variant}_{representation}'/'stock_day_predictions.parquet'
        else:
            path=args.root/'results'/'regression'/f'{prompt}_{args.model}_{args.variant}_{representation}'/'stock_day_predictions.stock_day_predictions.parquet'
        d=pd.read_parquet(path)[['stock_id','entry_date','actual_return','prediction']].rename(columns={'prediction':prompt}); d.entry_date=pd.to_datetime(d.entry_date); tables.append(d)
    merged=tables[0]
    for d in tables[1:]:
        # Actual returns are panel data, not a prompt-specific feature.  Join
        # on the stock-day key and validate that independent runs agree.
        merged=merged.merge(d,on=['stock_id','entry_date'],how='inner',validate='one_to_one',suffixes=('','_other'))
        if not np.allclose(merged.actual_return, merged.actual_return_other, equal_nan=True):
            raise ValueError('actual_return mismatch across prompt prediction files')
        merged=merged.drop(columns=['actual_return_other'])
    specs={p:merged[p] for p in PROMPTS}; specs['numeric_equal']=merged[list(PROMPTS)].mean(axis=1); specs['rank_equal']=merged[list(PROMPTS)].apply(rank_series).mean(axis=1)
    rows=[]
    for name,score in specs.items():
        daily=[]
        for year,g in merged.assign(score=score).groupby(merged.entry_date.dt.year):
            ic=g.score.rank().corr(g.actual_return.rank(),method='pearson')
            l=portfolio(g,g.score,0.20,'long')
            ls=portfolio(g,g.score,0.20,'long_short')
            rows.append({'model':args.model,'variant':args.variant,'representation':representation_label,'method':name,'year':int(year),'rows':len(g),'rankic':ic, 'long_gross':l['gross'],'long_net':l['net'],'long_gross_excess':l['gross_excess'],'long_net_excess':l['net_excess'],'long_cost':l['cost'],'long_positive_days':l['positive_days'],'ls_gross':ls['gross'],'ls_net':ls['net'],'ls_cost':ls['cost']})
    out=pd.DataFrame(rows); args.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False); overall=out.groupby('method',as_index=False).agg(years=('year','nunique'),rankic=('rankic','mean'),long_gross=('long_gross','mean'),long_net=('long_net','mean'),long_gross_excess=('long_gross_excess','mean'),long_net_excess=('long_net_excess','mean'),positive_years=('long_net',lambda s:int((s>0).sum())),positive_excess_years=('long_net_excess',lambda s:int((s>0).sum())),ls_net=('ls_net','mean')); overall.to_csv(args.output.with_name(args.output.stem+'_overall.csv'),index=False); print(overall.to_string(index=False))
if __name__=='__main__': main()
