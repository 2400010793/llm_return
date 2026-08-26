"""Leakage-safe UMAP+HDBSCAN versus PCA Ridge on Sina token matrices.

Selection uses validation top-20% long-short stock-day return only.  RankIC is
deliberately not used for model selection.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ALPHAS = (1.0, 10.0, 100.0, 1000.0)
MIN_CLUSTER = (25, 50, 100, 250)
MIN_SAMPLES = (3, 5, 10, 25)

def daily(frame, pred, target):
    x = frame[["stock_id", "entry_date", target]].copy()
    x["prediction"] = pred
    x["entry_date"] = pd.to_datetime(x["entry_date"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["stock_id", "entry_date", target, "prediction"])
    x = x.groupby(["stock_id", "entry_date"], as_index=False).agg(
        prediction=("prediction", "mean"), actual=(target, "mean"))
    return x

def top20_ls(d):
    vals = []
    for _, g in d.groupby("entry_date"):
        g = g.dropna(subset=["prediction", "actual"])
        if len(g) < 10: continue
        n = max(1, int(np.ceil(len(g) * .2)))
        s = g.sort_values("prediction")
        vals.append((s.tail(n).actual.mean() - s.head(n).actual.mean(),
                     s.tail(n).actual.mean(), s.head(n).actual.mean()))
    if not vals: return (-np.inf, np.nan, np.nan, 0)
    a = np.asarray(vals, float)
    return float(np.nanmean(a[:,0])), float(np.nanmean(a[:,1])), float(np.nanmean(a[:,2])), len(a)

def fit_space(matrix, idx, sample_rows, seed=42, mode="umap"):
    idx = np.asarray(idx, int)
    fit_idx = idx if len(idx) <= sample_rows else np.sort(np.random.default_rng(seed).choice(idx, sample_rows, replace=False))
    scaler = StandardScaler().fit(np.asarray(matrix[fit_idx], dtype=np.float32))
    pca = PCA(n_components=min(64, matrix.shape[1], len(fit_idx)-1), svd_solver="randomized", random_state=seed).fit(
        scaler.transform(np.asarray(matrix[fit_idx], dtype=np.float32)))
    if mode == "umap":
        reducer = umap.UMAP(n_components=8, n_neighbors=30, min_dist=.1, metric="cosine", random_state=seed,
                            transform_seed=seed, low_memory=True, n_jobs=1).fit(
                                pca.transform(scaler.transform(np.asarray(matrix[fit_idx], dtype=np.float32))))
    elif mode == "pca":
        reducer = None
    else:
        raise ValueError(f"unsupported mode: {mode}")
    def transform(pos):
        z = pca.transform(scaler.transform(np.asarray(matrix[np.asarray(pos, int)], dtype=np.float32)))
        return z.astype(np.float32) if reducer is None else reducer.transform(z).astype(np.float32)
    return scaler, pca, reducer, transform

def hdb_features(clusterer, z):
    labels, strengths = hdbscan.approximate_predict(clusterer, z)
    labels = labels.astype(int)
    labels[labels < 0] = clusterer.labels_.max() + 1
    k = int(clusterer.labels_.max() + 2)
    one = np.zeros((len(z), k), dtype=np.float32)
    one[np.arange(len(z)), np.clip(labels, 0, k-1)] = 1.0
    return np.hstack([z, one, strengths[:, None].astype(np.float32)]), labels, strengths

def run(args):
    panel = pd.read_parquet(args.panel)
    meta = pd.read_parquet(args.metadata)
    matrix = np.load(args.matrix, mmap_mode="r")
    order = pd.Series(np.arange(len(meta)), index=meta.row_index.astype(int))
    pos = panel.row_index.map(order)
    keep = pos.notna().to_numpy()
    frame = panel.loc[keep].reset_index(drop=True)
    matrix = matrix[pos[keep].astype(int).to_numpy()]
    target = args.target
    years = pd.to_datetime(frame.entry_date, errors="coerce").dt.year.to_numpy()
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
    test_year = args.test_year
    fit = np.flatnonzero(np.isfinite(y) & np.isin(years, np.arange(test_year-8, test_year-2)))
    val = np.flatnonzero(np.isfinite(y) & np.isin(years, [test_year-2, test_year-1]))
    all_train = np.flatnonzero(np.isfinite(y) & np.isin(years, np.arange(test_year-8, test_year)))
    test = np.flatnonzero(np.isfinite(y) & (years == test_year))
    _, _, _, transform = fit_space(matrix, fit, args.sample_rows, mode=args.mode)
    zfit, zval = transform(fit), transform(val)
    base_best=(-np.inf, None, None); cluster_best=(-np.inf, None, None, None)
    if args.fixed_min_cluster_size is not None:
        # Final fixed-configuration evaluation: validation is retained only
        # for audit rows, never for parameter selection.
        fixed_cl = hdbscan.HDBSCAN(min_cluster_size=args.fixed_min_cluster_size,
                                    min_samples=args.fixed_min_samples,
                                    metric="euclidean", prediction_data=True).fit(zfit)
        if len(np.unique(fixed_cl.labels_[fixed_cl.labels_ >= 0])) < 2:
            cluster_best = (-np.inf, args.fixed_alpha, None, None)
        else:
            cluster_best = (0.0, args.fixed_alpha, None, fixed_cl)
        base_best = (0.0, args.fixed_alpha, None)
    else:
      
      for alpha in ALPHAS:
          m=Ridge(alpha=alpha).fit(zfit,y[fit]); score=top20_ls(daily(frame.iloc[val],m.predict(zval),target))[0]
          if score>base_best[0]: base_best=(score,alpha,m)
      for mcs in MIN_CLUSTER:
          for ms in MIN_SAMPLES:
              cl=hdbscan.HDBSCAN(min_cluster_size=mcs,min_samples=ms,metric="euclidean",prediction_data=True).fit(zfit)
              if len(np.unique(cl.labels_[cl.labels_ >= 0])) < 2:
                  continue
              xf,_,_=hdb_features(cl,zfit); xv,_,_=hdb_features(cl,zval)
              for alpha in ALPHAS:
                  m=Ridge(alpha=alpha).fit(xf,y[fit]); score=top20_ls(daily(frame.iloc[val],m.predict(xv),target))[0]
                  if score>cluster_best[0]: cluster_best=(score,alpha,m,cl)
    # Refit preprocessing on train+validation, then freeze for test.
    _, _, _, transform_all = fit_space(matrix, all_train, args.sample_rows, mode=args.mode)
    za, zt = transform_all(all_train), transform_all(test)
    base=Ridge(alpha=base_best[1]).fit(za,y[all_train]); pred_base=base.predict(zt)
    if cluster_best[3] is None:
        # Explicit fallback for windows without a stable density partition.
        cluster_best = (-np.inf, 100.0, None, None)
        cl = None
    else:
        cl=hdbscan.HDBSCAN(min_cluster_size=cluster_best[3].min_cluster_size,min_samples=cluster_best[3].min_samples,
                           metric="euclidean",prediction_data=True).fit(za)
    if cl is None:
        xa, xt = za, zt
        labels = np.full(len(zt), -1, dtype=int)
        strength = np.zeros(len(zt), dtype=np.float32)
    else:
        xa,_,_=hdb_features(cl,za); xt,labels,strength=hdb_features(cl,zt)
    enhanced=Ridge(alpha=cluster_best[1]).fit(xa,y[all_train]); pred_cluster=enhanced.predict(xt)
    out=args.output_root/str(test_year); out.mkdir(parents=True,exist_ok=True)
    out_frame=frame.iloc[test][["row_index","stock_id","entry_date",target]].copy()
    out_frame["prediction_pca_ridge"]=pred_base; out_frame["prediction_umap_hdbscan"]=pred_cluster
    out_frame["hdbscan_cluster"]=labels; out_frame["hdbscan_strength"]=strength
    out_frame.to_parquet(out/"predictions.parquet",index=False)
    metrics={"test_year":test_year,"rows":{"fit":len(fit),"validation":len(val),"all_train":len(all_train),"test":len(test)},
      "validation_top20_ls":{"pca_ridge":base_best[0],"umap_hdbscan":cluster_best[0]},
      "selected":{"pca_alpha":base_best[1],"hdbscan_alpha":cluster_best[1],"min_cluster_size":None if cl is None else cl.min_cluster_size,"min_samples":None if cl is None else cl.min_samples,"fallback_pca":cl is None},
      "test_top20":{"pca_ridge":top20_ls(daily(out_frame,pred_base,target))[0],"umap_hdbscan":top20_ls(daily(out_frame,pred_cluster,target))[0]},
      "space_mode": args.mode,
      "cluster_counts":pd.Series(labels).value_counts().to_dict()}
    (out/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metrics,ensure_ascii=False))

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--panel",type=Path,required=True); p.add_argument("--matrix",type=Path,required=True); p.add_argument("--metadata",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--test-year",type=int,required=True); p.add_argument("--target",default="next_day_return"); p.add_argument("--sample-rows",type=int,default=20000); p.add_argument("--mode",choices=("umap","pca"),default="umap"); p.add_argument("--fixed-min-cluster-size",type=int); p.add_argument("--fixed-min-samples",type=int,default=10); p.add_argument("--fixed-alpha",type=float,default=1.0); run(p.parse_args())
