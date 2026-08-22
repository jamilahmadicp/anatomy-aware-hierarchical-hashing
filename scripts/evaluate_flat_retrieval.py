#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from anatomy_hash.data.manifest import load_manifest, split_df
from anatomy_hash.data.dataset import RadiographDataset
from anatomy_hash.data.transforms import build_transform
from anatomy_hash.checkpoints import load_flat_hash
from anatomy_hash.indexing import pack_codes, hamming_distance_packed
from anatomy_hash.metrics import retrieval_metrics_for_query
from anatomy_hash.training import device_from_arg

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--index",required=True); ap.add_argument("--out-dir",required=True); ap.add_argument("--query-split",default="test"); ap.add_argument("--relevance-col",default="fine_id"); ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--device",default="auto"); a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); z=np.load(a.index,allow_pickle=True); idx={k:z[k] for k in z.files}; df,_=load_manifest(a.manifest); qdf=split_df(df,a.query_split); model,ck=load_flat_hash(a.checkpoint,device_from_arg(a.device)); dl=DataLoader(RadiographDataset(qdf,build_transform(train=False)),batch_size=a.batch_size,num_workers=a.workers)
    rows=[]; off=0
    with torch.no_grad():
        for b in dl:
            q=model(b["image"].to(next(model.parameters()).device))["continuous_code"].sign().cpu().numpy().astype(np.int8); q[q==0]=1; qp=pack_codes(q)
            for i in range(len(q)):
                r=qdf.iloc[off+i]; dist=hamming_distance_packed(qp[i],idx["packed"],int(idx["n_bits"])); order=np.argsort(dist); rel=[]; total=len(set(str(idx["sample_id"][j]) for j in range(len(idx["sample_id"])) if str(idx["sample_id"][j])!=str(r.sample_id) and int(idx[a.relevance_col][j])==int(r[a.relevance_col])))
                seen=set()
                for j in order:
                    sid=str(idx["sample_id"][j]);
                    if sid==str(r.sample_id) or sid in seen: continue
                    seen.add(sid); rel.append(int(idx[a.relevance_col][j])==int(r[a.relevance_col]))
                m=retrieval_metrics_for_query(rel,total,(10,20,50,100,200)); m["sample_id"]=str(r.sample_id); rows.append(m)
            off+=len(q)
    rdf=pd.DataFrame(rows); rdf.to_csv(out/"query_metrics.csv",index=False); summary={"mAP":rdf.ap.mean(),**{c:rdf[c].mean() for c in rdf.columns if c.startswith(("p@","r@","ndcg@"))}}; pd.DataFrame([summary]).to_csv(out/"retrieval_summary.csv",index=False); print(summary)
if __name__=="__main__": main()
