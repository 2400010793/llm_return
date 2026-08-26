"""Build independent rolling classification/regression tasks for one dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True); p.add_argument('--dataset',required=True)
    p.add_argument('--prompts',default='profit,excess_return,return,loss')
    p.add_argument('--panel',type=Path,required=True); p.add_argument('--matrix-root',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--representations',default='prompt_mean,return_span')
    p.add_argument('--skip-missing',action='store_true',help='Omit prompt/representation combinations without a built matrix')
    args=p.parse_args(); rows=[]; representations=tuple(x.strip() for x in args.representations.split(',') if x.strip())
    prompts=tuple(x.strip() for x in args.prompts.split(',') if x.strip())
    allowed={'profit','excess_return','return','loss'}
    if not prompts or not set(prompts).issubset(allowed):
      raise ValueError(f'unsupported prompt: {prompts}')
    for prompt in prompts:
      for model in ('roberta','bge_m3'):
       for variant in ('short','masked_short'):
        for rep in representations:
         matrix=args.matrix_root/args.dataset/model/prompt/variant/rep
         if args.skip_missing and not (matrix/'matrix.npy').is_file():
          continue
         embed_model='bge_m3' if model=='bge_m3' else 'chinese_roberta'
         tag=f'{prompt}_{model}_{variant}_{rep}'
         for target in ('next_day_return','event_return_3d'):
          rows.append({'kind':'classification','dataset':args.dataset,'prompt':prompt,'model':model,'variant':variant,'representation':rep,'target':target,'panel':str(args.panel),'matrix':str(matrix/'matrix.npy'),'metadata':str(matrix/'metadata.parquet'),'embed_model':embed_model,'output':str(args.output_root/'classification'/tag/target)})
         rows.append({'kind':'regression','dataset':args.dataset,'prompt':prompt,'model':model,'variant':variant,'representation':rep,'target':'next_day_return','panel':str(args.panel),'matrix':str(matrix/'matrix.npy'),'metadata':str(matrix/'metadata.parquet'),'embed_model':embed_model,'output':str(args.output_root/'regression'/tag)})
    args.output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(args.output,sep='\t',index=False); print(f'wrote {len(rows)} tasks')

if __name__=='__main__': main()
