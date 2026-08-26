"""Write independent prompt comparison tables without cross-prompt fitting."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); args=p.parse_args(); s=args.root/'summary'
    cls=pd.read_csv(s/'prompt_classification_yearly.csv')
    stats=[]
    for f in (s/'simple_states/results').glob('*/'+'*.stats.csv'):
        stats.append(pd.read_csv(f).drop_duplicates('factor'))
    bt=pd.concat(stats,ignore_index=True) if stats else pd.DataFrame()
    cls.to_csv(s/'prompt_yearly_accuracy.csv',index=False)
    bt.to_csv(s/'prompt_yearly_backtest.csv',index=False)
    ls=pd.read_csv(s/'prompt_long_short_yearly.csv'); ls.to_csv(s/'prompt_long_short_decomposition.csv',index=False)
    # Paired annual bootstrap on the gross IC.  The comparison is descriptive;
    # no prompt is selected or refit from these test-year values.
    pair_rows=[]
    if not bt.empty and {'factor','IC'}.issubset(bt):
      bt['prompt']=bt.factor.str.extract(r'positive_(profit|excess_return|return|loss)_')[0]
      bt['year']=np.arange(len(bt)) % 9
      for metric in ('IC','Ret','RetL','RetS','Sharpe'):
       for a in ('profit','excess_return','return','loss'):
        for b in ('profit','excess_return','return','loss'):
         if a>=b: continue
         aa=bt[bt.prompt==a].groupby('year')[metric].mean(); bb=bt[bt.prompt==b].groupby('year')[metric].mean(); common=aa.index.intersection(bb.index)
         if len(common)<3: continue
         delta=(aa.loc[common]-bb.loc[common]).to_numpy(float); rng=np.random.default_rng(42); samples=rng.choice(delta,(2000,len(delta)),replace=True).mean(axis=1)
         pair_rows.append({'metric':metric,'prompt_a':a,'prompt_b':b,'mean_delta':float(delta.mean()),'bootstrap_ci_low':float(np.quantile(samples,.025)),'bootstrap_ci_high':float(np.quantile(samples,.975)),'years':len(delta)})
    pd.DataFrame(pair_rows).to_csv(s/'prompt_pairwise_bootstrap.csv',index=False)
    summary_lines=['# 四个 Prompt 独立实验汇报','', '本报告四个 prompt 分别训练、分别回测，未进行 prompt 拼接或联合选择。','']
    if not bt.empty:
      top=bt.sort_values('IC',ascending=False).drop_duplicates('factor').head(10)
      summary_lines += ['## 正式回测候选','',top[['factor','IC','ICIR','Ret','RetL','RetS','Sharpe','SharpeTN','ErrorCode']].to_markdown(index=False),'']
      summary_lines += ['`ErrorCode=1002` 和 `SharpeTN` 缺失仍需平台侧修复；当前只解释 IC、ICIR 和毛收益。','']
    if not cls.empty:
      topc=cls.groupby(['tag','target'],as_index=False).accuracy.mean().sort_values('accuracy',ascending=False).head(10)
      summary_lines += ['## 分类平均 Accuracy','',topc.to_markdown(index=False),'']
    (s/'report_zh.md').write_text('\n'.join(summary_lines),encoding='utf-8')
    print(json.dumps({'report':str(s/'report_zh.md'),'bootstrap_rows':len(pair_rows),'backtest_rows':len(bt)},ensure_ascii=False))

if __name__=='__main__': main()
