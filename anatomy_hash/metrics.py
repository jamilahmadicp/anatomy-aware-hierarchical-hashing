from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


def classification_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
    }


def multiclass_auc_metrics(y_true, prob, n_classes):
    y_bin = np.eye(n_classes, dtype=int)[np.asarray(y_true, dtype=int)]
    out = {}
    for avg in ["micro", "macro"]:
        try: out[f"{avg}_auroc"] = float(roc_auc_score(y_bin, prob, average=avg, multi_class="ovr"))
        except ValueError: out[f"{avg}_auroc"] = float("nan")
        try: out[f"{avg}_auprc"] = float(average_precision_score(y_bin, prob, average=avg))
        except ValueError: out[f"{avg}_auprc"] = float("nan")
    return out


def expected_calibration_error(y_true, prob, n_bins=15):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    details = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi if hi < 1 else conf <= hi)
        if not m.any(): continue
        acc = correct[m].mean(); c = conf[m].mean(); w = m.mean()
        ece += w * abs(acc - c)
        details.append((float(lo), float(hi), float(acc), float(c), int(m.sum())))
    return float(ece), details


def average_precision_from_ranked(relevance, total_relevant=None):
    rel = np.asarray(relevance, dtype=np.int32)
    denom = int(rel.sum()) if total_relevant is None else int(total_relevant)
    if denom <= 0: return np.nan
    prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
    return float((prec * rel).sum() / denom)


def ndcg_at_k(relevance, k):
    rel = np.asarray(relevance, dtype=float)[:k]
    if rel.size == 0: return np.nan
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = ((2**rel - 1) * discounts).sum()
    ideal = np.sort(np.asarray(relevance, dtype=float))[::-1][:k]
    idcg = ((2**ideal - 1) * discounts[:len(ideal)]).sum()
    return float(dcg / idcg) if idcg > 0 else np.nan


def retrieval_metrics_for_query(relevance, total_relevant, ks=(10,20,50,100,200)):
    rel = np.asarray(relevance, dtype=int)
    out = {"ap": average_precision_from_ranked(rel, total_relevant)}
    for k in ks:
        kk = min(k, len(rel))
        out[f"p@{k}"] = float(rel[:kk].mean()) if kk else np.nan
        out[f"r@{k}"] = float(rel[:kk].sum() / total_relevant) if total_relevant > 0 else np.nan
        out[f"ndcg@{k}"] = ndcg_at_k(rel, k)
    return out
