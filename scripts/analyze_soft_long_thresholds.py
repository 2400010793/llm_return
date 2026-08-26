"""Descriptive long-only threshold sensitivity for completed soft-family folds."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--fold-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    rows=[]
    for path in sorted(args.fold_root.glob('*/*/*/*/stock_day_predictions.parquet')):
        part=pd.read_parquet(path); part['entry_date']=pd.to_datetime(part['entry_date'])
        # Thresholds are cross-sectional fractions; this is a sensitivity audit,
        # not a test-year parameter selection procedure.
        for q in (0.50,0.70,0.80,0.85,0.90):
            for year,daypart in part.groupby(part.entry_date.dt.year):
                vals=[]
                for _,g in daypart.groupby('entry_date'):
                    if len(g)<5: continue
                    cutoff=g.prediction.quantile(q)
                    top=g[g.prediction>=cutoff]
                    if len(top): vals.append(float(top.actual_return.mean()))
                if vals:
                    rows.append({'model':path.parents[3].name,'feature_mode':path.parents[2].name,'target':path.parents[1].name,'test_year':int(year),'top_fraction':1-q,'mean_daily_long_return':sum(vals)/len(vals),'positive_days':sum(x>0 for x in vals)/len(vals),'days':len(vals)})
    out=pd.DataFrame(rows); args.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False)
    overall=out.groupby(['model','feature_mode','target','top_fraction'],as_index=False).agg(mean_daily_long_return=('mean_daily_long_return','mean'),positive_days=('positive_days','mean'),years=('test_year','nunique'))
    overall.to_csv(args.output.with_name(args.output.stem+'_overall.csv'),index=False)
    print(overall.to_string(index=False))
if __name__=='__main__': main()
