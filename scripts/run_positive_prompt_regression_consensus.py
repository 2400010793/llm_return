"""Evaluate four-prompt consensus on independently produced Ridge forecasts."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

PROMPTS=("profit","excess_return","return","loss")
TAG_RE=re.compile(r"^(profit|excess_return|return|loss)_(bge_m3|roberta)_(short|masked_short)_(prompt_mean|full_mean|return_span|stock_span)$")

def rank_frame(d, name):
    d=d.copy(); d['entry_date']=pd.to_datetime(d['entry_date']); d[name]=d.groupby('entry_date')['prediction'].rank(pct=True); return d[['stock_id','entry_date','actual_return',name]]

def join_paths(paths, validation):
    frames=[]
    for prompt,path in paths.items():
        if validation:
            ps=sorted((path/'artifacts').glob('test_year_*/validation_stock_day_predictions.parquet'))
            d=pd.concat([pd.read_parquet(x) for x in ps],ignore_index=True)
        else: d=pd.read_parquet(path/'stock_day_predictions.stock_day_predictions.parquet')
        d=d.drop_duplicates(['stock_id','entry_date'],keep='last')
        frames.append(rank_frame(d,prompt))
    out=frames[0]
    for d in frames[1:]: out=out.merge(d,on=['stock_id','entry_date','actual_return'],how='inner')
    return out

def score(d, threshold, votes):
    vals=d[list(PROMPTS)].to_numpy(float); long=(vals>=threshold).sum(1)>=votes; short=(vals<=1-threshold).sum(1)>=votes; sig=np.where(long,1,np.where(short,-1,0)); actual=np.sign(d.actual_return.to_numpy(float)); sel=sig!=0
    return {'score':float(np.mean(sig[sel]==actual[sel])) if sel.any() else -np.inf,'coverage':float(sel.mean()),'long_hit':float(np.mean(actual[sig==1]>0)) if (sig==1).any() else np.nan,'short_hit':float(np.mean(actual[sig==-1]<0)) if (sig==-1).any() else np.nan}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); args=p.parse_args(); base=args.root/'results'/'regression'; groups={}
    for pred in base.glob('*/stock_day_predictions.stock_day_predictions.parquet'):
        m=TAG_RE.match(pred.parent.name)
        if m: groups.setdefault(m.groups()[1:],{})[m.groups()[0]]=pred.parent
    out=args.root/'summary'/'consensus'; out.mkdir(parents=True,exist_ok=True); rows=[]
    for key,paths in sorted(groups.items()):
        if set(paths)!=set(PROMPTS): continue
        val=join_paths(paths,True); test=join_paths(paths,False); best=None
        for threshold in (.55,.60,.65,.70):
          for votes in (2,3,4):
            s=score(val,threshold,votes)
            if best is None or s['score']>best[0]: best=(s['score'],threshold,votes)
        _,threshold,votes=best; s=score(test,threshold,votes); vals=test[list(PROMPTS)].to_numpy(float); lv=(vals>=threshold).sum(1); sv=(vals<=1-threshold).sum(1); test['consensus_rank']=vals.mean(1); test['signal']=np.where(lv>=votes,1,np.where(sv>=votes,-1,0)); name='_'.join(key); test.to_parquet(out/f'regression_{name}_predictions.parquet',index=False); rows.append({'model':key[0],'variant':key[1],'representation':key[2],'threshold':threshold,'votes':votes,'validation_score':best[0],**s,'test_rows':len(test)})
    pd.DataFrame(rows).to_csv(out/'regression_consensus_metrics.csv',index=False); print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
