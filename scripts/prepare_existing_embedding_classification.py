"""Align completed full-corpus embeddings to the labeled CNINFO panel."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd

def key(x): return re.sub(r'\s+','',str(x or ''))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--panel',type=Path,required=True); p.add_argument('--cleaned',type=Path,required=True); p.add_argument('--embedding',type=Path,required=True); p.add_argument('--metadata',type=Path,required=True); p.add_argument('--out-panel',type=Path,required=True); p.add_argument('--out-npy',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
 panel=pd.read_parquet(a.panel).copy(); meta=pd.read_parquet(a.metadata); matrix=np.load(a.embedding,mmap_mode='r')
 panel['_sid']=panel.stock_id.astype(str).str.zfill(6); panel['_titlek']=panel.title.map(key)
 panel['_date']=pd.to_datetime(panel.published_at,errors='coerce')
 needed=set(zip(panel['_sid'],panel['_titlek']))
 candidates={}
 with a.cleaned.open('r',encoding='utf-8') as fh:
    for row_number,line in enumerate(fh):
     record=json.loads(line)
     candidate_key=(str(record.get('stock_id','')).zfill(6),key(record.get('title','')))
     if candidate_key not in needed: continue
     date=pd.to_datetime(record.get('announcement_date'),errors='coerce')
     if pd.isna(date): continue
     candidates.setdefault(candidate_key,[]).append((date,row_number))
 rows=[]; unmatched=0
 for idx,r in panel.iterrows():
    matches=candidates.get((r._sid,r._titlek),[])
    if not matches: unmatched+=1; continue
    hit_date,hit_row=min(matches,key=lambda item: abs(item[0]-r._date))
    rows.append((idx,int(hit_row),float(abs((hit_date-r._date).days))))
 rows_df=pd.DataFrame(rows,columns=['panel_index','embedding_row','date_distance_days'])
 rows_df=rows_df.drop_duplicates('embedding_row').drop_duplicates('panel_index')
 out=panel.loc[rows_df.panel_index.to_numpy()].copy().reset_index(drop=True); out['embedding_row']=rows_df.embedding_row.to_numpy(); out['alignment_date_distance_days']=rows_df.date_distance_days.to_numpy()
 a.out_panel.parent.mkdir(parents=True,exist_ok=True); a.out_npy.parent.mkdir(parents=True,exist_ok=True)
 np.save(a.out_npy,np.asarray(matrix[rows_df.embedding_row.to_numpy()],dtype=np.float32)); out.to_parquet(a.out_panel,index=False)
 report={'panel_rows':len(panel),'matched_rows':len(out),'unmatched_rows':unmatched,'embedding_rows':len(matrix),'max_date_distance_days':float(rows_df.date_distance_days.max()) if len(rows_df) else None,'median_date_distance_days':float(rows_df.date_distance_days.median()) if len(rows_df) else None}
 a.report.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report))
if __name__=='__main__': main()
