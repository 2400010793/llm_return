"""Summarize independent positive-prompt results and build backtest manifest."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); args=p.parse_args()
    out=args.root/'summary'; out.mkdir(parents=True,exist_ok=True)
    cls=[]; back=[]; longshort=[]
    for report in sorted((args.root/'results'/'classification').glob('*/*/artifacts/report.json')):
        d=json.loads(report.read_text()); tag=report.parents[2].name; target=report.parents[1].name
        for row in d.get('results',[]):
            cls.append({'tag':tag,'target':target,**row})
    # The classifier reports one row per test year.  Keep prompt fields explicit.
    cdf=pd.DataFrame(cls)
    if not cdf.empty:
        for col in ('tag',):
            parts=cdf[col].str.split('_',n=4,expand=True)
        cdf.to_csv(out/'prompt_classification_yearly.csv',index=False)
    for pred in sorted((args.root/'results'/'regression').glob('*/stock_day_predictions.stock_day_predictions.parquet')):
        tag=pred.parent.name; d=pd.read_parquet(pred); d['entry_date']=pd.to_datetime(d['entry_date'])
        d['year']=d['entry_date'].dt.year
        for year,g in d.groupby('year'):
            if len(g)<10: continue
            g=g.sort_values('prediction'); n=max(1,len(g)//5)
            low=g.head(n); high=g.tail(n)
            longshort.append({'tag':tag,'year':int(year),'rows':len(g),'long_rows':len(high),'short_rows':len(low),'long_return':float(high.actual_return.mean()),'short_return':float(-low.actual_return.mean()),'long_short':float(high.actual_return.mean()-low.actual_return.mean()),'long_hit_rate':float((high.actual_return>0).mean()),'short_hit_rate':float((low.actual_return<0).mean())})
        # simple_states reads this exact long-format stock-day file.
        back.append({'predictions':str(pred),'factor_id':'positive_'+tag,'prediction_column':'prediction'})
    pd.DataFrame(longshort).to_csv(out/'prompt_long_short_yearly.csv',index=False)
    manifest=pd.DataFrame(back); manifest.insert(0,'task_id',range(len(manifest))); manifest.to_csv(out/'simple_states_manifest.tsv',sep='\t',index=False)
    (out/'summary.json').write_text(json.dumps({'classification_configs':int(cdf[['tag','target']].drop_duplicates().shape[0]) if not cdf.empty else 0,'regression_configs':len(back),'backtest_manifest':str(out/'simple_states_manifest.tsv')},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'classification_rows':len(cdf),'regression_configs':len(back),'manifest':str(out/'simple_states_manifest.tsv')},ensure_ascii=False))

if __name__=='__main__': main()
