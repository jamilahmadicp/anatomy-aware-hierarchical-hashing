#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from anatomy_hash.data.manifest import load_manifest, split_df
from anatomy_hash.classical import lsh_fit_transform, spectral_hashing_pca, klsh_anchor_rbf
from anatomy_hash.indexing import pack_codes, hamming_distance_packed
from anatomy_hash.metrics import retrieval_metrics_for_query

def evaluate(qcodes,dbcodes,qdf,dbdf,relevance):
    qp=pack_codes(qcodes); dp=pack_codes(dbcodes); bits=qcodes.shape[1]; rows=[]
    for i,r in qdf.iterrows():
        dist=hamming_distance_packed(qp[i],dp,bits); order=np.argsort(dist); total=int((dbdf[relevance].values==int(r[relevance])).sum()); rel=(dbdf.iloc[order][relevance].values==int(r[relevance])).astype(int); rows.append(retrieval_metrics_for_query(rel,total,(20,50,100,200)))
    x=pd.DataFrame(rows); return {"mAP":x.ap.mean(),**{c:x[c].mean() for c in x.columns if c.startswith(("p@","r@"))}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--representations",required=True,help="NPZ from extract_search_representations for database split"); ap.add_argument("--query-representations",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--db-splits",nargs="+",default=["train","val"]); ap.add_argument("--query-split",default="test"); ap.add_argument("--bits",nargs="+",type=int,default=[32,64,128,256]); ap.add_argument("--relevance-col",default="fine_id"); ap.add_argument("--out",required=True); ap.add_argument("--seed",type=int,default=42); a=ap.parse_args()
    dbz=np.load(a.representations,allow_pickle=True); qz=np.load(a.query_representations,allow_pickle=True); xdb=dbz["features"]; xq=qz["features"]; df,_=load_manifest(a.manifest); dbdf=split_df(df,a.db_splits); qdf=split_df(df,a.query_split); allx=np.concatenate([xdb,xq]); rows=[]
    for bits in a.bits:
        for name,fn in [("LSH",lsh_fit_transform),("SH-PCA",spectral_hashing_pca),("KLSH-anchor",klsh_anchor_rbf)]:
            codes=fn(xdb,allx,bits=bits,seed=a.seed); met=evaluate(codes[len(xdb):],codes[:len(xdb)],qdf,dbdf,a.relevance_col); rows.append({"method":name,"bits":bits,**met}); print(rows[-1])
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.out,index=False)
if __name__=="__main__": main()
