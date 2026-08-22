#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from anatomy_hash.data.manifest import load_manifest, split_df

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--predictions",required=True,help="stage2_predictions.npz from evaluate_stage2_classification.py"); ap.add_argument("--split",default="val"); ap.add_argument("--out-dir",required=True); a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); df,maps=load_manifest(a.manifest); d=split_df(df,a.split); z=np.load(a.predictions,allow_pickle=True); prob=z["prob"]; sids=z["sample_id"].astype(str)
    # Align by sample_id robustly.
    pos_labels={int(fid) for label,fid in maps["fine"].items() if str(label).lower().endswith("positive")}
    ppos=prob[:,sorted(pos_labels)].sum(axis=1)
    pred_df=pd.DataFrame({"sample_id":sids,"p_abnormal":ppos}); x=d.merge(pred_df,on="sample_id",how="inner")
    if x.study_id.eq("").all(): raise SystemExit("MURA study_id missing; rebuild manifest with prepare_mura_manifest.py")
    study=x.groupby("study_id",as_index=False).agg(y=("abnormal_label","first"),p=("p_abnormal","mean"))
    y=study.y.to_numpy(int); p=study.p.to_numpy(float); pred=(p>=0.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    met={"n_studies":len(study),"auroc":roc_auc_score(y,p),"auprc":average_precision_score(y,p),"accuracy":accuracy_score(y,pred),"f1":f1_score(y,pred),"sensitivity":tp/max(1,tp+fn),"specificity":tn/max(1,tn+fp),"cohen_kappa":cohen_kappa_score(y,pred)}
    pd.DataFrame([met]).to_csv(out/"mura_binary_study_metrics.csv",index=False); study.to_csv(out/"mura_binary_study_predictions.csv",index=False)
    fpr,tpr,_=roc_curve(y,p); pr,rc,_=precision_recall_curve(y,p)
    fig,ax=plt.subplots(figsize=(6,5)); ax.plot(fpr,tpr,label=f"AUROC={met['auroc']:.3f}"); ax.plot([0,1],[0,1],"--"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(); fig.tight_layout(); fig.savefig(out/"mura_binary_roc.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); ax.plot(rc,pr,label=f"AUPRC={met['auprc']:.3f}"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(); fig.tight_layout(); fig.savefig(out/"mura_binary_pr.png",dpi=300); plt.close(fig)
    print(met)
if __name__=="__main__": main()
