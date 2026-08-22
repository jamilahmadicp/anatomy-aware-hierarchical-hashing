#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from anatomy_hash.data.manifest import load_manifest, split_df
from anatomy_hash.data.dataset import RadiographDataset
from anatomy_hash.data.transforms import build_transform
from anatomy_hash.models.stage1 import Stage1AnatomyClassifier
from anatomy_hash.training import class_weights, make_optimizer, make_scheduler, device_from_arg, amp_context
from anatomy_hash.utils.seed import seed_everything
from anatomy_hash.utils.io import ensure_dir, environment_snapshot, save_json

def eval_model(model,loader,device):
    model.eval(); ys=[]; ps=[]; loss=[]; ce=nn.CrossEntropyLoss()
    with torch.no_grad():
        for b in loader:
            x=b["image"].to(device); y=b["anatomy"].to(device); lg=model(x); loss.append(float(ce(lg,y))); ys.extend(y.cpu().tolist()); ps.extend(lg.argmax(1).cpu().tolist())
    return np.mean(loss), float((np.asarray(ys)==np.asarray(ps)).mean())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--model",default="convnext_tiny.fb_in1k"); ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--weight-decay",type=float,default=0.05); ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--device",default="auto"); ap.add_argument("--patience",type=int,default=7); ap.add_argument("--no-amp",action="store_true")
    a=ap.parse_args(); seed_everything(a.seed); device=device_from_arg(a.device); df,maps=load_manifest(a.manifest)
    tr=split_df(df,"train"); va=split_df(df,"val"); n=int(df.anatomy_id.max()+1)
    train_ds=RadiographDataset(tr,build_transform(train=True)); val_ds=RadiographDataset(va,build_transform(train=False))
    train=DataLoader(train_ds,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=device.type=="cuda")
    val=DataLoader(val_ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=device.type=="cuda")
    model=Stage1AnatomyClassifier(n,a.model,True).to(device); w=class_weights(tr.anatomy_id,n).to(device); ce=nn.CrossEntropyLoss(weight=w)
    opt=make_optimizer(model,a.lr,a.weight_decay); sch=make_scheduler(opt,a.epochs); scaler=torch.amp.GradScaler("cuda", enabled=device.type=="cuda" and not a.no_amp)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); best=-1; bad=0; hist=[]
    for ep in range(a.epochs):
        model.train(); losses=[]
        for b in tqdm(train,desc=f"stage1 {ep+1}/{a.epochs}"):
            x=b["image"].to(device); y=b["anatomy"].to(device); opt.zero_grad(set_to_none=True)
            with amp_context(device,not a.no_amp): loss=ce(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
        sch.step(); vl,acc=eval_model(model,val,device); hist.append({"epoch":ep+1,"train_loss":float(np.mean(losses)),"val_loss":vl,"val_accuracy":acc})
        if acc>best:
            best=acc; bad=0
            torch.save({"state_dict":model.state_dict(),"model_name":a.model,"num_classes":n,"temperature":1.0,"seed":a.seed,"label_maps":maps},out)
        else: bad+=1
        if bad>=a.patience: break
    pd.DataFrame(hist).to_csv(out.with_suffix(".history.csv"),index=False); save_json(environment_snapshot(),out.with_suffix(".environment.json")); print(f"best val accuracy={best:.4f} -> {out}")
if __name__=="__main__": main()
