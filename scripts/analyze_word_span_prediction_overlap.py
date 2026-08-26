"""Replicate the reference prompt-factor prediction overlap analysis."""
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

PROMPTS={"profit":"profit_span","excess_return":"excess_span","return":"plain_return_span","loss":"loss_span"}

def corr_rows(frame: pd.DataFrame, left: str, right: str, group=None):
    vals=[]
    iterator=frame.groupby(group,sort=True) if group else [(None,frame)]
    for key,g in iterator:
        g=g[[left,right]].dropna()
        if len(g)<5 or g[left].nunique()<2 or g[right].nunique()<2: continue
        s=spearmanr(g[left],g[right]).statistic; p=pearsonr(g[left],g[right])[0]
        vals.append({"group":key,"spearman":float(s),"pearson":float(p),"rows":len(g)})
    return vals

def load(path,name):
    d=pd.read_parquet(path,columns=["stock_id","entry_date","prediction"])
    d["entry_date"]=pd.to_datetime(d.entry_date,errors="coerce").dt.normalize()
    d=d.dropna().drop_duplicates(["stock_id","entry_date"])
    return d.rename(columns={"prediction":name})

def run_group(specs, out, group_label):
    frames=[]
    for name,path in specs:
        if path.exists(): frames.append(load(path,name))
    if len(frames)<2: return None
    merged=frames[0]
    for d in frames[1:]: merged=merged.merge(d,on=["stock_id","entry_date"],how="inner",validate="one_to_one")
    names=[x[0] for x in specs if x[1].exists()]
    rows=[]; yearly=[]
    merged["year"]=merged.entry_date.dt.year
    for a,b in combinations(names,2):
        z=corr_rows(merged,a,b)[0]; rows.append({"group":group_label,"left":a,"right":b,"common_rows":len(merged),**{k:v for k,v in z.items() if k!="group"}})
        yr=corr_rows(merged,a,b,"year");
        for x in yr: yearly.append({"group":group_label,"left":a,"right":b,"test_year":int(x["group"]),"common_rows":x["rows"],"spearman":x["spearman"],"pearson":x["pearson"]})
    pd.DataFrame(rows).to_csv(out/f"{group_label}_pairwise.csv",index=False)
    pd.DataFrame(yearly).to_csv(out/f"{group_label}_yearly.csv",index=False)
    return merged

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--qwen-root',type=Path,required=True); ap.add_argument('--output-root',type=Path,required=True); args=ap.parse_args(); args.output_root.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for variant in ('short','masked_short'):
        for model in ('roberta','bge_m3'):
            specs=[(prompt,args.root/f"{prompt}_{model}_{variant}_{span}/stock_day_predictions.stock_day_predictions.parquet") for prompt,span in PROMPTS.items()]
            run_group(specs,args.output_root,f'{model}_{variant}')
        cross=[]
        for prompt,span in PROMPTS.items():
            cross.append((f"roberta_{prompt}",args.root/f"{prompt}_roberta_{variant}_{span}/stock_day_predictions.stock_day_predictions.parquet"))
            cross.append((f"bge_m3_{prompt}",args.root/f"{prompt}_bge_m3_{variant}_{span}/stock_day_predictions.stock_day_predictions.parquet"))
        # Cross-model same-prompt pairs only, matching the reference output.
        rows=[]; yearly=[]
        for prompt,span in PROMPTS.items():
            specs=[(f'roberta_{prompt}',args.root/f"{prompt}_roberta_{variant}_{span}/stock_day_predictions.stock_day_predictions.parquet"),(f'bge_m3_{prompt}',args.root/f"{prompt}_bge_m3_{variant}_{span}/stock_day_predictions.stock_day_predictions.parquet")]
            m=run_group(specs,args.output_root,f'cross_{prompt}_{variant}')
        qroot=args.qwen_root/'results'; qspec=[(f'qwen_{rep}',qroot/f'{variant}_{d}/stock_day_predictions.stock_day_predictions.parquet') for rep,d in [('article_mean','article_mean'),('prompt_mean','prompt_mean'),('return_token','return_token')]]
        run_group(qspec,args.output_root,f'qwen_{variant}')
    print(f'wrote prediction-overlap tables to {args.output_root}')

if __name__=='__main__': main()
