#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from anatomy_hash.data.manifest import load_manifest, split_df
from anatomy_hash.data.dataset import RadiographDataset
from anatomy_hash.data.transforms import build_transform
from anatomy_hash.checkpoints import load_stage1, load_stage2
from anatomy_hash.evaluation import calibrated_stage1_prob
from anatomy_hash.indexing import corrupt_probabilities, choose_routes, pack_codes
from anatomy_hash.retrieval import search_index, evaluate_ranked_query
from anatomy_hash.training import device_from_arg

def li(p):
    z=np.load(p,allow_pickle=True); return {k:z[k] for k in z.files}

def eval_policy(qdf, cache, idx, s2, device, error_rate, policy, top_r, alpha, relevance, seed):
    rng=np.random.default_rng(seed); rows=[]
    for i,r in qdf.iterrows():
        p=corrupt_probabilities(cache["prob"][i],error_rate,rng); true_a=int(r.anatomy_id); routes=choose_routes(p,policy,top_r,true_anatomy=true_a); feat=torch.from_numpy(cache["features"][i:i+1]).to(device); codes={}
        with torch.no_grad():
            for c in routes:
                q=s2.route_from_features(feat,torch.tensor([c],device=device))["continuous_code"].sign().cpu().numpy().astype(np.int8); q[q==0]=1; codes[c]=pack_codes(q)[0]
        ranked=search_index(idx,codes,p,top_r,policy,alpha,0,true_anatomy=true_a); m,_=evaluate_ranked_query(ranked,idx,r,relevance,(20,50)); rows.append(m)
    rdf=pd.DataFrame(rows); return float(rdf.ap.mean()),float(rdf["p@20"].mean()),float(rdf["r@20"].mean())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--stage1",required=True); ap.add_argument("--stage2",required=True); ap.add_argument("--index-top1",required=True); ap.add_argument("--index-adaptive",required=True); ap.add_argument("--out-dir",required=True); ap.add_argument("--query-split",default="test"); ap.add_argument("--relevance-col",default="fine_id"); ap.add_argument("--alpha",type=float,default=0.05); ap.add_argument("--rates",nargs="+",type=float,default=[0,.05,.10,.15,.20,.30]); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--device",default="auto"); ap.add_argument("--workers",type=int,default=8); a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); device=device_from_arg(a.device); df,_=load_manifest(a.manifest); qdf=split_df(df,a.query_split); s1,T,_=load_stage1(a.stage1,device); s2,_=load_stage2(a.stage2,device); dl=DataLoader(RadiographDataset(qdf,build_transform(train=False)),batch_size=64,num_workers=a.workers,shuffle=False)
    probs=[]; feats=[]
    with torch.no_grad():
        for b in dl:
            x=b["image"].to(device); probs.append(calibrated_stage1_prob(s1,x,T).cpu().numpy()); feats.append(s2.encode(x).cpu().numpy())
    cache={"prob":np.concatenate(probs),"features":np.concatenate(feats)}; idx1=li(a.index_top1); idxa=li(a.index_adaptive); rows=[]
    for e in a.rates:
        for label,idx,pol,tr in [("Top1 index + Top1 query",idx1,"top1",1),("Top1 index + Top2 query",idx1,"topk",2),("Adaptive index + Top2 query",idxa,"topk",2)]:
            m,p,r=eval_policy(qdf,cache,idx,s2,device,e,pol,tr,a.alpha,a.relevance_col,a.seed+int(e*1000)); rows.append({"injected_routing_error":e,"policy":label,"mAP":m,"p@20":p,"r@20":r}); print(rows[-1])
    rdf=pd.DataFrame(rows); rdf.to_csv(out/"routing_robustness.csv",index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    for pol,g in rdf.groupby("policy"): ax.plot(g.injected_routing_error*100,g.mAP,marker="o",label=pol)
    ax.set_xlabel("Injected Stage-1 routing error (%)"); ax.set_ylabel("mAP"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/"routing_robustness.png",dpi=300); plt.close(fig)
if __name__=="__main__": main()
