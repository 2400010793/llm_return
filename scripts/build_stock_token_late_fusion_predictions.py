"""Build aligned stock-day prediction files for fixed-weight late fusion."""
from pathlib import Path
import argparse
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--variant',choices=('short','masked_short'),required=True); p.add_argument('--weight',type=float,default=.5); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    parts=[]
    for year in range(2018,2027):
        base=a.root/'stock_focus/kmeans_ridge'
        r=pd.read_parquet(base/'roberta/short/stock_span'/a.variant/'next_day_return'/str(year)/'stock_day_predictions.parquet').rename(columns={'prediction':'roberta_prediction'})
        b=pd.read_parquet(base/'bge_m3/short/stock_span'/a.variant/'next_day_return'/str(year)/'stock_day_predictions.parquet').rename(columns={'prediction':'bge_prediction'})
        x=r.merge(b,on=['stock_id','entry_date'],suffixes=('','_bge'))
        x['prediction']=a.weight*x.roberta_prediction+(1-a.weight)*x.bge_prediction
        parts.append(x[['stock_id','entry_date','prediction','actual','roberta_prediction','bge_prediction']])
    out=pd.concat(parts,ignore_index=True); a.output_dir.mkdir(parents=True,exist_ok=True)
    out.to_parquet(a.output_dir/f'late_fusion_{a.variant}_{a.weight:.2f}.parquet',index=False)
    out[['stock_id','entry_date','roberta_prediction','actual']].rename(columns={'roberta_prediction':'prediction'}).to_parquet(a.output_dir/f'roberta_intersection_{a.variant}.parquet',index=False)
    out[['stock_id','entry_date','bge_prediction','actual']].rename(columns={'bge_prediction':'prediction'}).to_parquet(a.output_dir/f'bge_intersection_{a.variant}.parquet',index=False)
    print({'rows':len(out),'path':str(a.output_dir/f'late_fusion_{a.variant}_{a.weight:.2f}.parquet')})
if __name__=='__main__': main()
