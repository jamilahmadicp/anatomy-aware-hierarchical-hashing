#!/usr/bin/env python
from __future__ import annotations
import argparse, json, platform, time
from pathlib import Path
import numpy as np, pandas as pd
from anatomy_hash.indexing import pack_codes, hamming_distance_packed

def cosine_scan(q, db):
    q=q/(np.linalg.norm(q)+1e-12); d=db/(np.linalg.norm(db,axis=1,keepdims=True)+1e-12); return 1-d@q

def bench(fn,warmup,repeats):
    for _ in range(warmup): fn()
    times=[]
    for _ in range(repeats):
        t=time.perf_counter(); fn(); times.append((time.perf_counter()-t)*1000)
    return float(np.mean(times)),float(np.std(times,ddof=1) if len(times)>1 else 0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--representations",required=True); ap.add_argument("--out-dir",required=True); ap.add_argument("--sizes",nargs="+",type=int,default=[5000,15000,65000,130000]); ap.add_argument("--queries",type=int,default=100); ap.add_argument("--warmup",type=int,default=10); ap.add_argument("--repeats",type=int,default=20); ap.add_argument("--seed",type=int,default=42); a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); z=np.load(a.representations,allow_pickle=True); feat=z["features"].astype(np.float32); codes=z["codes"].astype(np.int8); anat=z["anatomy_id"].astype(int); bits=codes.shape[1]; rng=np.random.default_rng(a.seed); rows=[]
    qidx=rng.choice(len(feat),min(a.queries,len(feat)),replace=False); qfeat=feat[qidx]; qcodes=pack_codes(codes[qidx]); qanat=anat[qidx]
    for N in a.sizes:
        # Scalability stress test: resample actual indexed representations with replacement when N exceeds source size.
        ix=rng.choice(len(feat),N,replace=N>len(feat)); fdb=feat[ix]; cdb=pack_codes(codes[ix]); adb=anat[ix]
        vals={"float_cosine":[],"full_hamming":[],"routed_hamming":[]}
        for qf,qc,qa in zip(qfeat,qcodes,qanat):
            m=adb==qa
            for name,fn in [("float_cosine",lambda qf=qf: cosine_scan(qf,fdb)),("full_hamming",lambda qc=qc: hamming_distance_packed(qc,cdb,bits)),("routed_hamming",lambda qc=qc,m=m: hamming_distance_packed(qc,cdb[m],bits))]:
                mean,sd=bench(fn,a.warmup,a.repeats); vals[name].append(mean)
        row={"index_entries":N,"source_unique_images":len(feat),"construction":"resampled actual representations; replacement used if index_entries > source_unique_images"}
        for name,v in vals.items(): row[f"{name}_ms_per_query"]=float(np.mean(v)); row[f"{name}_sd_ms"]=float(np.std(v,ddof=1) if len(v)>1 else 0)
        rows.append(row); print(row)
    pd.DataFrame(rows).to_csv(out/"efficiency.csv",index=False)
    with open(out/"efficiency_protocol.json","w") as f: json.dump({"units":"milliseconds per query; search-only, excludes neural encoding and disk I/O","queries":len(qidx),"warmup":a.warmup,"repeats":a.repeats,"cpu":platform.processor(),"numpy":np.__version__},f,indent=2)
    import matplotlib.pyplot as plt
    rdf=pd.DataFrame(rows); fig,ax=plt.subplots(figsize=(7,5))
    for c,label in [("float_cosine_ms_per_query","Float cosine scan"),("full_hamming_ms_per_query","Full Hamming scan"),("routed_hamming_ms_per_query","Anatomy-routed Hamming")]: ax.plot(rdf.index_entries,rdf[c],marker="o",label=label)
    ax.set_xlabel("Indexed entries"); ax.set_ylabel("Search latency (ms/query)"); ax.legend(); fig.tight_layout(); fig.savefig(out/"efficiency.png",dpi=300); plt.close(fig)
if __name__=="__main__": main()
