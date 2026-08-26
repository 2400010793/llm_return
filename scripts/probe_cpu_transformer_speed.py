from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.text.embeddings import mean_pool

p=argparse.ArgumentParser()
p.add_argument('--model-path', required=True)
p.add_argument('--name', required=True)
p.add_argument('--rows', type=int, default=256)
p.add_argument('--batch-size', type=int, default=8)
p.add_argument('--max-length', type=int, default=256)
p.add_argument('--input', type=Path, required=True)
a=p.parse_args()
texts=[]
with a.input.open(encoding='utf-8') as f:
    for line in f:
        if line.strip():
            r=json.loads(line); texts.append(str(r.get('text_plain','')))
            if len(texts)>=a.rows: break
texts=texts[:a.rows]
t0=time.perf_counter()
tok=AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
model=AutoModel.from_pretrained(a.model_path, local_files_only=True).to('cpu').eval()
load=time.perf_counter()-t0
start=time.perf_counter(); n=0
with torch.inference_mode():
    for i in range(0,len(texts),a.batch_size):
        enc=tok(texts[i:i+a.batch_size],padding=True,truncation=True,max_length=a.max_length,return_tensors='pt')
        out=model(**enc)
        _=mean_pool(out.last_hidden_state,enc['attention_mask'])
        n += len(texts[i:i+a.batch_size])
elapsed=time.perf_counter()-start
print(json.dumps({'name':a.name,'rows':n,'load_seconds':load,'encode_seconds':elapsed,'total_seconds':load+elapsed,'rows_per_second':n/elapsed,'seconds_per_row':elapsed/n},ensure_ascii=False))
