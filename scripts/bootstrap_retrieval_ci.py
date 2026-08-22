#!/usr/bin/env python
from __future__ import annotations
import argparse
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--query-metrics",required=True); ap.add_argument("--out",required=True); ap.add_argument("--n-bootstrap",type=int,default=5000); ap.add_argument("--seed",type=int,default=42); a=ap.parse_args(); df=pd.read_csv(a.query_metrics); rng=np.random.default_rng(a.seed); cols=[c for c in df.columns if c=="ap" or c.startswith(("p@","r@","ndcg@"))]; rows=[]
    n=len(df)
    for c in cols:
        x=df[c].dropna().to_numpy(float); vals=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(a.n_bootstrap)])
        rows.append({"metric":"mAP" if c=="ap" else c,"mean":x.mean(),"ci95_lo":np.quantile(vals,.025),"ci95_hi":np.quantile(vals,.975),"n_queries":len(x)})
    pd.DataFrame(rows).to_csv(a.out,index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
