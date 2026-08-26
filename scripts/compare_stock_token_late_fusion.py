"""Evaluate fixed-weight late fusion of completed stock-token predictions."""
from pathlib import Path
import argparse
import pandas as pd

def ic(frame):
    vals=[]
    for _, g in frame.groupby("entry_date"):
        if len(g)>=5 and g.prediction.nunique()>1 and g.actual.nunique()>1:
            vals.append(g.prediction.corr(g.actual, method="spearman"))
    return float(pd.Series(vals).mean()) if vals else float("nan")

def load(root, model, variant, year):
    p=root/f"stock_focus/kmeans_ridge/{model}/short/stock_span/{variant}/next_day_return/{year}/stock_day_predictions.parquet"
    x=pd.read_parquet(p).rename(columns={"prediction":f"pred_{model}"})
    return x[["stock_id","entry_date",f"pred_{model}","actual"]]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--variant",choices=("short","masked_short"),default="short"); a=ap.parse_args()
    rows=[]
    for year in range(2018,2027):
        x=load(a.root,"roberta",a.variant,year); y=load(a.root,"bge_m3",a.variant,year)
        z=x.merge(y,on=["stock_id","entry_date"],suffixes=("","_bge"))
        for w in (0.0,0.25,0.5,0.75,1.0):
            q=z[["stock_id","entry_date","actual"]].copy(); q["prediction"]=w*z.pred_roberta+(1-w)*z.pred_bge_m3
            rows.append({"year":year,"variant":a.variant,"roberta_weight":w,"rows":len(q),"ic":ic(q)})
    out=pd.DataFrame(rows); a.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
    print(out.groupby("roberta_weight").apply(lambda g: (g.ic*g.rows).sum()/g.rows.sum()).to_string())

if __name__=="__main__": main()
