#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from anatomy_hash.data.manifest import load_manifest, split_df
from anatomy_hash.data.dataset import RadiographDataset
from anatomy_hash.data.transforms import build_transform
from anatomy_hash.checkpoints import load_stage1
from anatomy_hash.training import device_from_arg
from anatomy_hash.metrics import classification_metrics, multiclass_auc_metrics, expected_calibration_error
from anatomy_hash.plotting import save_confusion, save_micro_roc_pr, save_reliability_diagram

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--split",default="test"); ap.add_argument("--out-dir",required=True); ap.add_argument("--batch-size",type=int,default=128); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--device",default="auto"); a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); device=device_from_arg(a.device); df,maps=load_manifest(a.manifest); d=split_df(df,a.split); ds=RadiographDataset(d,build_transform(train=False)); dl=DataLoader(ds,batch_size=a.batch_size,num_workers=a.workers)
    model,T,ck=load_stage1(a.checkpoint,device); ys=[]; probs=[]; sids=[]
    with torch.no_grad():
        for b in dl:
            p=F.softmax(model(b["image"].to(device))/T,1).cpu().numpy(); probs.append(p); ys.extend(b["anatomy"].numpy()); sids.extend(b["sample_id"])
    prob=np.concatenate(probs); y=np.asarray(ys); pred=prob.argmax(1); met=classification_metrics(y,pred); met.update(multiclass_auc_metrics(y,prob,prob.shape[1])); met["ece"],details=expected_calibration_error(y,prob)
    pd.DataFrame([met]).to_csv(out/"stage1_metrics.csv",index=False); np.savez_compressed(out/"stage1_predictions.npz",y_true=y,prob=prob,sample_id=np.array(sids,dtype=object))
    names=[v for v,_ in sorted(maps["anatomy"].items(),key=lambda kv:int(kv[1]))]; save_confusion(y,pred,names,out/"stage1_confusion.png",True,"Stage-1 anatomy classification")
    save_micro_roc_pr(y,prob,prob.shape[1],out/"stage1")
    save_reliability_diagram(details,out/"stage1_reliability.png","Stage-1 calibration")
    print(met)
if __name__=="__main__": main()
