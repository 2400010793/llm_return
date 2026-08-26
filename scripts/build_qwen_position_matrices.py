"""Materialize completed Qwen Sina prompt means with an optional position filter."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--embedding-root',type=Path,required=True); p.add_argument('--panel',type=Path,required=True); p.add_argument('--prompt',default='return'); p.add_argument('--variant',required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--fixed-position',type=int,default=None); p.add_argument('--groups',default='all,fixed')
    a=p.parse_args(); base=a.embedding_root/'sina'/a.prompt/a.variant; leaves=sorted(base.glob('shard-*/COMPLETED'))
    if not leaves: raise SystemExit('no completed shards')
    panel=pd.read_parquet(a.panel).set_index('row_index',drop=False)
    counts=[]; max_pos=-1; dim=None
    for done in leaves:
      d=done.parent; emb=np.load(d/'prompt_token_embeddings.npy',mmap_mode='r'); meta=[json.loads(x) for x in (d/'metadata.jsonl').open(encoding='utf-8')]
      if len(meta)!=emb.shape[0]: raise ValueError(f'mismatch {d}')
      dim=emb.shape[2]; max_pos=max(max_pos,max(int(x['prompt_start_zero_based']) for x in meta)); counts.append((d,emb.shape[0],meta))
    fixed=a.fixed_position if a.fixed_position is not None else max_pos
    keep_rows=[]; all_rows=[]
    for d,n,meta in counts:
      for i,m in enumerate(meta):
        if int(m['row_index']) in panel.index:
          all_rows.append((d,i,m))
          if int(m['prompt_start_zero_based'])==fixed: keep_rows.append((d,i,m))
    wanted=set(a.groups.split(',')); groups=[]
    if 'all' in wanted: groups.append(('all_positions',all_rows))
    if 'fixed' in wanted: groups.append(('fixed_position',keep_rows))
    for name,selected in groups:
      out=a.output_root/name; out.mkdir(parents=True,exist_ok=True); matrix=np.lib.format.open_memmap(out/'matrix.npy',mode='w+',dtype='float32',shape=(len(selected),dim)); rows=[]
      for j,(d,i,m) in enumerate(selected): matrix[j]=np.asarray(np.load(d/'prompt_token_embeddings.npy',mmap_mode='r')[i].mean(axis=0),dtype=np.float32); rows.append(int(m['row_index']))
      del matrix
      pf=panel.loc[rows].reset_index(drop=True); pf.to_parquet(out/'panel.parquet',index=False); pd.DataFrame({'row_index':rows,'prompt_start_zero_based':[int(m['prompt_start_zero_based']) for _,_,m in selected]}).to_parquet(out/'metadata.parquet',index=False)
      (out/'summary.json').write_text(json.dumps({'prompt':a.prompt,'variant':a.variant,'fixed_position':fixed,'rows':len(rows),'dimension':dim},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'shards':len(leaves),'fixed_position':fixed,'all_rows':len(all_rows),'fixed_rows':len(keep_rows),'dimension':dim},ensure_ascii=False))
if __name__=='__main__': main()
