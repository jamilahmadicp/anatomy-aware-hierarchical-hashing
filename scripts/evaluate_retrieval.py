#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from anatomy_hash.data.manifest import load_manifest, split_df
from anatomy_hash.data.dataset import RadiographDataset
from anatomy_hash.data.transforms import build_transform
from anatomy_hash.checkpoints import load_stage1, load_stage2
from anatomy_hash.evaluation import calibrated_stage1_prob
from anatomy_hash.indexing import choose_routes, pack_codes
from anatomy_hash.retrieval import search_index, evaluate_ranked_query
from anatomy_hash.training import device_from_arg
from anatomy_hash.plotting import save_retrieval_grid


def load_index(path):
    z=np.load(path,allow_pickle=True); return {k:z[k] for k in z.files}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--stage1",required=True); ap.add_argument("--stage2",required=True); ap.add_argument("--index",required=True); ap.add_argument("--out-dir",required=True)
    ap.add_argument("--query-split",default="test"); ap.add_argument("--query-policy",choices=["top1","topk","adaptive","oracle"],default="topk"); ap.add_argument("--top-r",type=int,default=2); ap.add_argument("--threshold",type=float,default=0.8); ap.add_argument("--alpha",type=float,default=0.05); ap.add_argument("--relevance-col",choices=["fine_id","joint_id","anatomy_id"],default="fine_id"); ap.add_argument("--max-k",type=int,default=0); ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--device",default="auto"); ap.add_argument("--qualitative",type=int,default=8); a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); device=device_from_arg(a.device); df,_=load_manifest(a.manifest); qdf=split_df(df,a.query_split); dl=DataLoader(RadiographDataset(qdf,build_transform(train=False)),batch_size=a.batch_size,num_workers=a.workers,shuffle=False)
    idx=load_index(a.index); s1,T,_=load_stage1(a.stage1,device); s2,ck2=load_stage2(a.stage2,device); results=[]; examples=[]; offset=0
    with torch.no_grad():
        for b in dl:
            x=b["image"].to(device); pa=calibrated_stage1_prob(s1,x,T).cpu().numpy(); feat=s2.encode(x)
            for i in range(len(x)):
                r=qdf.iloc[offset+i]; true_a=int(r.anatomy_id); routes=choose_routes(pa[i],a.query_policy,a.top_r,a.threshold,true_a); codes={}
                for c in routes:
                    o=s2.route_from_features(feat[i:i+1],torch.tensor([c],device=device)); code=o["continuous_code"].sign().cpu().numpy().astype(np.int8); code[code==0]=1; codes[c]=pack_codes(code)[0]
                ranked=search_index(idx,codes,pa[i],a.top_r,a.query_policy,a.alpha,a.max_k,true_anatomy=true_a,threshold=a.threshold)
                met,filtered=evaluate_ranked_query(ranked,idx,r,a.relevance_col,ks=(10,20,50,100,200)); met.update({"sample_id":str(r.sample_id),"true_anatomy":true_a,"top1_anatomy":int(pa[i].argmax())}); results.append(met)
                if len(examples)<a.qualitative:
                    exres=[]
                    for sid,sc,d,row in filtered[:10]:
                        exres.append({"sample_id":sid,"path":str(idx["path"][row]),"distance":float(d),"score":float(sc),"result_label":int(idx[a.relevance_col][row]),"correct":int(idx[a.relevance_col][row])==int(r[a.relevance_col])})
                    examples.append({"query_sample_id":str(r.sample_id),"query_path":str(r.path),"query_label":int(r[a.relevance_col]),"results":exres})
            offset += len(x)
    rdf=pd.DataFrame(results); rdf.to_csv(out/"query_metrics.csv",index=False); summary={"n_queries":len(rdf),"mAP":float(rdf.ap.mean())}
    for c in [x for x in rdf.columns if x.startswith(("p@","r@","ndcg@"))]: summary[c]=float(rdf[c].mean())
    pd.DataFrame([summary]).to_csv(out/"retrieval_summary.csv",index=False)
    with open(out/"retrieval_examples.json","w") as f: json.dump(examples,f,indent=2)
    if examples: save_retrieval_grid(examples,out/"retrieval_examples.png",10,"Qualitative top-10 retrieval examples")
    print(summary)
if __name__=="__main__": main()
