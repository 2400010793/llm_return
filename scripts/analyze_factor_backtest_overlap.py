"""Backtest and Top20 overlap comparison for body/span/Qwen representations."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROMPTS = {"profit":"profit_span", "excess_return":"excess_span", "return":"plain_return_span", "loss":"loss_span"}

def load_factor(path: Path, name: str) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=["stock_id","entry_date","actual_return","prediction"])
    d["entry_date"] = pd.to_datetime(d.entry_date, errors="coerce").dt.normalize()
    d = d.dropna(subset=["stock_id","entry_date","actual_return","prediction"]).drop_duplicates(["stock_id","entry_date"])
    return d.rename(columns={"prediction": name})

def daily_portfolio(d: pd.DataFrame, score: str) -> dict[str,float]:
    rows=[]
    for _,g in d.groupby("entry_date", sort=False):
        if len(g)<10: continue
        n=max(1,int(np.ceil(len(g)*.2))); s=g.sort_values([score,"stock_id"],kind="mergesort")
        lo=float(s.head(n).actual_return.mean()); hi=float(s.tail(n).actual_return.mean())
        rows.append((hi,lo,hi-lo))
    if not rows: return {"days":0,"long_bp":np.nan,"short_leg_bp":np.nan,"ls_bp":np.nan}
    a=np.asarray(rows); return {"days":len(a),"long_bp":a[:,0].mean()*1e4,"short_leg_bp":a[:,1].mean()*1e4,"ls_bp":a[:,2].mean()*1e4}

def top_sets(d: pd.DataFrame, score: str) -> dict[pd.Timestamp,set[str]]:
    out={}
    for day,g in d.groupby("entry_date",sort=False):
        if len(g)<10: continue
        n=max(1,int(np.ceil(len(g)*.2))); out[day]=set(g.sort_values([score,"stock_id"],kind="mergesort").tail(n).stock_id.astype(str))
    return out

def overlap(a: dict, b: dict) -> dict[str,float]:
    common=sorted(set(a)&set(b)); vals=[]; recall=[]
    for day in common:
        x,y=a[day],b[day]; inter=len(x&y); union=len(x|y)
        vals.append(inter/union if union else np.nan); recall.append(inter/min(len(x),len(y)) if min(len(x),len(y)) else np.nan)
    return {"common_days":len(common),"mean_jaccard":float(np.nanmean(vals)) if vals else np.nan,"mean_overlap_rate":float(np.nanmean(recall)) if recall else np.nan}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--sina-root",type=Path,required=True); ap.add_argument("--qwen-root",type=Path,required=True); ap.add_argument("--output-root",type=Path,required=True); ap.add_argument("--variant",default="masked_short"); args=ap.parse_args(); args.output_root.mkdir(parents=True,exist_ok=True)
    specs=[]
    for model in ("roberta","bge_m3"):
        for prompt,span in PROMPTS.items():
            for rep,dirn in (("body_mean","full_mean"),("word_span",span)):
                p=args.sina_root/f"{prompt}_{model}_{args.variant}_{dirn}"/"stock_day_predictions.stock_day_predictions.parquet"
                if p.exists(): specs.append((f"{model}__{prompt}__{rep}",model,prompt,rep,p))
    qroot=args.qwen_root/"results"
    for rep,dirn in (("article_mean_body_proxy","article_mean"),("prompt_mean","prompt_mean"),("return_token","return_token")):
        p=qroot/f"{args.variant}_{dirn}"/"stock_day_predictions.stock_day_predictions.parquet"
        if p.exists(): specs.append((f"qwen3_embedding_8b__return__{rep}","qwen3_embedding_8b","return",rep,p))
    frames=[]; meta=[]
    for name,model,prompt,rep,p in specs:
        d=load_factor(p,name); frames.append(d); meta.append({"factor":name,"model":model,"prompt":prompt,"representation":rep,"path":str(p),"rows":len(d)})
    merged=frames[0]
    for d in frames[1:]:
        merged=merged.merge(d,on=["stock_id","entry_date","actual_return"],how="inner",validate="one_to_one")
    names=[x[0] for x in specs]; sets={n:top_sets(merged,n) for n in names}
    rows=[]
    for name,model,prompt,rep,p in specs:
        m=daily_portfolio(merged,name); rows.append({"factor":name,"model":model,"prompt":prompt,"representation":rep,"common_rows":len(merged),**m})
    perf=pd.DataFrame(rows); perf.to_csv(args.output_root/"factor_backtest_metrics.csv",index=False)
    ov=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            x=overlap(sets[a],sets[b]); ov.append({"factor_left":a,"factor_right":b,**x})
    pd.DataFrame(ov).to_csv(args.output_root/"factor_top20_overlap.csv",index=False)
    within=[]
    for model in ("roberta","bge_m3"):
        for prompt in PROMPTS:
            b=f"{model}__{prompt}__body_mean"; w=f"{model}__{prompt}__word_span"
            if b in sets and w in sets:
                z=overlap(sets[b],sets[w]); bp=perf.set_index("factor"); within.append({"model":model,"prompt":prompt,"body_factor":b,"word_factor":w,"delta_long_bp":bp.loc[w].long_bp-bp.loc[b].long_bp,"delta_short_leg_bp":bp.loc[w].short_leg_bp-bp.loc[b].short_leg_bp,"delta_ls_bp":bp.loc[w].ls_bp-bp.loc[b].ls_bp,**z})
    pd.DataFrame(within).to_csv(args.output_root/"word_span_vs_body_backtest.csv",index=False)
    q=[n for n in names if n.startswith("qwen3_embedding_8b")]
    qrows=[]
    for i,a in enumerate(q):
        for b in q[i+1:]: qrows.append({"left":a,"right":b,**overlap(sets[a],sets[b])})
    pd.DataFrame(qrows).to_csv(args.output_root/"qwen_representation_overlap.csv",index=False)
    pd.DataFrame(meta).to_json(args.output_root/"overlap_audit.json",orient="records",force_ascii=False,indent=2)
    print(perf.to_string(index=False)); print("common_rows",len(merged),"factors",len(names))
if __name__=="__main__": main()
