"""Evaluate rule-based consensus of four independently trained prompts.

No model is refit here and no prompt embeddings are combined.  Probabilities
are joined by row key only after each prompt has produced its own OOS output.
Thresholds and vote counts are selected on validation rows, then frozen for
test years.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

PROMPTS=("profit","excess_return","return","loss")
TAG_RE=re.compile(r"^(profit|excess_return|return|loss)_(bge_m3|roberta)_(short|masked_short)_(prompt_mean|full_mean|return_span|stock_span)$")

def _tag(path: Path):
    m=TAG_RE.match(path.parent.parent.name)
    return m.groups() if m else None

def _joined(paths, validation=False):
    frames=[]
    for prompt,path in paths.items():
        if validation:
            parts=[]
            for p in sorted(path.parent.joinpath('artifacts').glob('test_year_*/validation_predictions.parquet')):
                parts.append(pd.read_parquet(p))
            d=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
        else:
            p=path/'predictions.predictions.parquet'; d=pd.read_parquet(p)
        if d.empty: return d
        d=d.drop_duplicates(['row_index','test_year'],keep='last')
        d=d[['row_index','entry_date','next_day_return','test_year','probability']].rename(columns={'probability':prompt})
        frames.append(d)
    out=frames[0]
    for d in frames[1:]: out=out.merge(d[['row_index',PROMPTS[[p for p in PROMPTS if p in d.columns][0]] if False else d.columns[-1]]],on='row_index',how='inner')
    return out

def _join(paths, validation):
    frames=[]
    for prompt,path in paths.items():
        if validation:
            ps=sorted((path/'artifacts').glob('test_year_*/validation_predictions.parquet'))
            d=pd.concat([pd.read_parquet(p) for p in ps],ignore_index=True)
        else: d=pd.read_parquet(path/'predictions.predictions.parquet')
        d=d.drop_duplicates(['row_index','test_year'],keep='last')
        d=d[['row_index','entry_date','next_day_return','test_year','probability']].rename(columns={'probability':prompt})
        frames.append(d)
    out=frames[0]
    for d in frames[1:]: out=out.merge(d,on=['row_index','entry_date','next_day_return','test_year'],how='inner')
    return out

def _score(d, threshold, votes):
    p=d[list(PROMPTS)].to_numpy(float); mean=p.mean(axis=1)
    long_votes=(p>=0.5).sum(axis=1); short_votes=(p<0.5).sum(axis=1)
    signal=np.where((mean>=threshold)&(long_votes>=votes),1,np.where((mean<=1-threshold)&(short_votes>=votes),-1,0))
    actual=np.sign(d['next_day_return'].to_numpy(float))
    selected=signal!=0
    if not selected.any(): return {'score':-np.inf,'coverage':0.0,'long_hit':np.nan,'short_hit':np.nan}
    long=signal==1; short=signal==-1
    return {'score':float(np.mean(signal[selected]==actual[selected])),'coverage':float(selected.mean()),'long_hit':float(np.mean(actual[long]>0)) if long.any() else np.nan,'short_hit':float(np.mean(actual[short]<0)) if short.any() else np.nan}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); args=p.parse_args()
    base=args.root/'results'/'classification'; groups={}
    for pred in base.glob('*/next_day_return/predictions.predictions.parquet'):
        info=_tag(pred)
        if not info: continue
        prompt,model,variant,rep=info; groups.setdefault((model,variant,rep),{})[prompt]=pred.parent
    out=args.root/'summary'/'consensus'; out.mkdir(parents=True,exist_ok=True); metrics=[]
    for (model,variant,rep),paths in sorted(groups.items()):
        if set(paths)!=set(PROMPTS): continue
        val=_join(paths,True); test=_join(paths,False)
        best=None
        for threshold in (.55,.60,.65,.70):
          for votes in (2,3,4):
            scored=_score(val,threshold,votes)
            if best is None or scored['score']>best[0]: best=(scored['score'],threshold,votes,scored)
        _,threshold,votes,_=best; scored=_score(test,threshold,votes)
        pvals=test[list(PROMPTS)].to_numpy(float); mean=pvals.mean(1); lv=(pvals>=.5).sum(1); sv=(pvals<.5).sum(1)
        test['consensus_probability']=mean; test['signal']=np.where((mean>=threshold)&(lv>=votes),1,np.where((mean<=1-threshold)&(sv>=votes),-1,0))
        test.to_parquet(out/f'{model}_{variant}_{rep}_predictions.parquet',index=False)
        metrics.append({'model':model,'variant':variant,'representation':rep,'threshold':threshold,'votes':votes,'validation_score':best[0],**scored,'test_rows':len(test)})
    pd.DataFrame(metrics).to_csv(out/'consensus_metrics.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False))

if __name__=='__main__': main()
