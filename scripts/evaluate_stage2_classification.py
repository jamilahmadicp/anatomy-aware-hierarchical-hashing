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
from anatomy_hash.training import device_from_arg
from anatomy_hash.evaluation import hierarchical_global_prob
from anatomy_hash.metrics import classification_metrics, multiclass_auc_metrics
from anatomy_hash.plotting import save_confusion, save_micro_roc_pr, save_tsne

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--stage1",required=True); ap.add_argument("--stage2",required=True); ap.add_argument("--split",default="test"); ap.add_argument("--top-r",type=int,default=2); ap.add_argument("--out-dir",required=True); ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--device",default="auto"); a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); device=device_from_arg(a.device); df,maps=load_manifest(a.manifest); d=split_df(df,a.split); dl=DataLoader(RadiographDataset(d,build_transform(train=False)),batch_size=a.batch_size,num_workers=a.workers)
    s1,T,_=load_stage1(a.stage1,device); s2,_=load_stage2(a.stage2,device); ys=[]; probs=[]; embeds=[]; sids=[]
    with torch.no_grad():
        for b in dl:
            x=b["image"].to(device); pa,pf,feat=hierarchical_global_prob(s1,s2,x,T,a.top_r); probs.append(pf.cpu().numpy()); ys.extend(b["fine"].numpy()); embeds.append(feat.cpu().numpy()); sids.extend(b["sample_id"])
    p=np.concatenate(probs); y=np.asarray(ys); e=np.concatenate(embeds); pred=p.argmax(1); met=classification_metrics(y,pred); met.update(multiclass_auc_metrics(y,p,p.shape[1]))
    pd.DataFrame([met]).to_csv(out/"stage2_metrics.csv",index=False); np.savez_compressed(out/"stage2_predictions.npz",y_true=y,prob=p,embedding=e,sample_id=np.array(sids,dtype=object))
    names=[v for v,_ in sorted(maps["fine"].items(),key=lambda kv:int(kv[1]))]
    save_confusion(y,pred,names,out/"stage2_confusion.png",True,"Fine-grained classification")
    if len(names)<=20: save_micro_roc_pr(y,p,p.shape[1],out/"stage2")
    save_tsne(e,y,names,out/"stage2_tsne.png")
    print(met)
if __name__=="__main__": main()
