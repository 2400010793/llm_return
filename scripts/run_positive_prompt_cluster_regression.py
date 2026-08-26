"""Independent rolling PCA + MiniBatchKMeans Ridge for one prompt config."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def stock_day(frame,pred):
 d=frame[['stock_id','entry_date','next_day_return']].copy(); d['prediction']=pred; return d.groupby(['stock_id','entry_date'],as_index=False).agg(actual_return=('next_day_return','mean'),prediction=('prediction','mean'))
def ic(frame,pred):
 d=stock_day(frame,pred); vals=[]
 for _,g in d.groupby('entry_date'):
  if len(g)>=5 and g.prediction.nunique()>1 and g.actual_return.nunique()>1: vals.append(g.prediction.corr(g.actual_return,method='spearman'))
 return float(np.nanmean(vals)) if vals else -np.inf
def transform(matrix,idx,pca,scaler): return scaler.transform(pca.transform(np.asarray(matrix[idx],dtype=np.float32)))
def main():
 p=argparse.ArgumentParser(); p.add_argument('panel',type=Path); p.add_argument('--matrix',type=Path,required=True); p.add_argument('--metadata',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--components',type=int,default=128); p.add_argument('--seed',type=int,default=42); args=p.parse_args()
 frame=pd.read_parquet(args.panel); matrix=np.load(args.matrix,mmap_mode='r'); meta=pd.read_parquet(args.metadata)
 if len(frame)!=len(matrix) or not np.array_equal(frame.row_index.to_numpy(),meta.row_index.to_numpy()): raise ValueError('matrix metadata panel row alignment failed')
 frame['entry_date']=pd.to_datetime(frame.entry_date); years=sorted(frame.loc[frame.entry_date.dt.year>=2010,'entry_date'].dt.year.unique()); returns=pd.to_numeric(frame.next_day_return,errors='coerce').to_numpy(float); outputs=[]; audit=[]
 for pos in range(8,len(years)):
  test_year=years[pos]; train_years=years[pos-8:pos-2]; val_years=years[pos-2:pos]; all_years=years[pos-8:pos]
  y=frame.entry_date.dt.year.to_numpy(); fit=np.flatnonzero(np.isin(y,train_years)&np.isfinite(returns)); val=np.flatnonzero(np.isin(y,val_years)&np.isfinite(returns)); all_idx=np.flatnonzero(np.isin(y,all_years)&np.isfinite(returns)); test=np.flatnonzero((y==test_year)&np.isfinite(returns))
  pca=PCA(n_components=min(args.components,matrix.shape[1],len(fit)-1),svd_solver='randomized',random_state=args.seed).fit(np.asarray(matrix[fit],dtype=np.float32)); scaler=StandardScaler().fit(pca.transform(np.asarray(matrix[fit],dtype=np.float32))); xfit=transform(matrix,fit,pca,scaler); xv=transform(matrix,val,pca,scaler)
  best=(-np.inf,2,1000.)
  for k in (2,4,6,8,12):
   km=MiniBatchKMeans(n_clusters=k,random_state=args.seed,batch_size=2048,n_init=3,max_iter=200).fit(xfit); zf=np.eye(k,dtype=np.float32)[km.predict(xfit)]; zv=np.eye(k,dtype=np.float32)[km.predict(xv)]
   for alpha in (1.,10.,100.,1000.):
    model=Ridge(alpha=alpha).fit(np.hstack([xfit,zf]),returns[fit]); score=ic(frame.iloc[val],model.predict(np.hstack([xv,zv])))
    if score>best[0]: best=(score,k,alpha)
  pca=PCA(n_components=min(args.components,matrix.shape[1],len(all_idx)-1),svd_solver='randomized',random_state=args.seed).fit(np.asarray(matrix[all_idx],dtype=np.float32)); scaler=StandardScaler().fit(pca.transform(np.asarray(matrix[all_idx],dtype=np.float32))); xa=transform(matrix,all_idx,pca,scaler); xt=transform(matrix,test,pca,scaler); km=MiniBatchKMeans(n_clusters=int(best[1]),random_state=args.seed,batch_size=2048,n_init=3,max_iter=200).fit(xa); model=Ridge(alpha=float(best[2])).fit(np.hstack([xa,np.eye(int(best[1]))[km.predict(xa)]]),returns[all_idx]); pred=model.predict(np.hstack([xt,np.eye(int(best[1]))[km.predict(xt)]])); d=frame.iloc[test][['stock_id','entry_date','next_day_return']].copy(); d['prediction']=pred; d['test_year']=test_year; outputs.append(stock_day(d,pred).assign(test_year=test_year)); audit.append({'test_year':int(test_year),'selected_k':int(best[1]),'alpha':float(best[2]),'validation_ic':float(best[0])})
 result=pd.concat(outputs,ignore_index=True); args.output.parent.mkdir(parents=True,exist_ok=True); result.to_parquet(args.output,index=False); args.output.with_suffix('.metrics.json').write_text(json.dumps({'rows':len(result),'years':audit},ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
