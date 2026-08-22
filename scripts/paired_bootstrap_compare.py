#!/usr/bin/env python
from __future__ import annotations
import argparse
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--proposed",required=True); ap.add_argument("--baseline",required=True); ap.add_argument("--out",required=True); ap.add_argument("--metric",default="ap"); ap.add_argument("--n-bootstrap",type=int,default=10000); ap.add_argument("--seed",type=int,default=42); a=ap.parse_args(); p=pd.read_csv(a.proposed)[["sample_id",a.metric]].rename(columns={a.metric:"p"}); b=pd.read_csv(a.baseline)[["sample_id",a.metric]].rename(columns={a.metric:"b"}); d=p.merge(b,on="sample_id").dropna(); diff=(d.p-d.b).to_numpy(float); rng=np.random.default_rng(a.seed); boot=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(a.n_bootstrap)]); obs=diff.mean(); pval=2*min((boot<=0).mean(),(boot>=0).mean()); out=pd.DataFrame([{"metric":a.metric,"mean_difference":obs,"ci95_lo":np.quantile(boot,.025),"ci95_hi":np.quantile(boot,.975),"bootstrap_two_sided_p":min(1.0,pval),"n_queries":len(diff)}]); out.to_csv(a.out,index=False); print(out.to_string(index=False))
if __name__=="__main__": main()
