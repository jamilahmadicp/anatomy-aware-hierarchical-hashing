from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
from PIL import Image


def save_confusion(y_true, y_pred, labels, path, normalize=True, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)), normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, aspect="auto")
    ax.set_title(title); ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    if len(labels) <= 20:
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def save_micro_roc_pr(y_true, prob, n_classes, prefix):
    y = np.eye(n_classes)[np.asarray(y_true, dtype=int)]
    fpr, tpr, _ = roc_curve(y.ravel(), prob.ravel()); roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y.ravel(), prob.ravel()); pr_auc = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(7,5)); ax.plot(fpr, tpr, label=f"Micro ROC (AUC={roc_auc:.3f})"); ax.plot([0,1],[0,1], "--")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(); fig.tight_layout(); fig.savefig(str(prefix)+"_roc.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,5)); ax.plot(rec, prec, label=f"Micro PR (AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(); fig.tight_layout(); fig.savefig(str(prefix)+"_pr.png", dpi=300); plt.close(fig)
    return roc_auc, pr_auc


def save_tsne(embeddings, labels, label_names, path, max_points=5000, seed=42):
    x = np.asarray(embeddings); y = np.asarray(labels)
    if len(x) > max_points:
        rng = np.random.default_rng(seed); idx = rng.choice(len(x), max_points, replace=False); x=x[idx]; y=y[idx]
    z = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(x)
    fig, ax = plt.subplots(figsize=(9,7))
    for c in np.unique(y):
        m=y==c; ax.scatter(z[m,0], z[m,1], s=8, alpha=.65, label=label_names[int(c)] if int(c)<len(label_names) else str(c))
    if len(np.unique(y)) <= 20: ax.legend(fontsize=6, bbox_to_anchor=(1.02,1), loc="upper left")
    ax.set_xlabel("t-SNE dimension 1"); ax.set_ylabel("t-SNE dimension 2"); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def save_retrieval_grid(examples, path, n_results=10, title=None):
    rows = len(examples); cols = n_results + 1
    fig, axes = plt.subplots(rows, cols, figsize=(2.0*cols, 2.4*rows), squeeze=False)
    for r, ex in enumerate(examples):
        items = [(ex["query_path"], f"Query\nClass {ex.get('query_label','?')}", True, True)]
        items += [(v["path"], f"Match {k+1}\nClass {v.get('result_label','?')}", bool(v["correct"]), False) for k,v in enumerate(ex["results"][:n_results])]
        for c in range(cols):
            ax=axes[r,c]; ax.axis("off")
            if c>=len(items): continue
            pth, lab, correct, is_query = items[c]
            with Image.open(pth) as im: ax.imshow(im.convert("L"), cmap="gray")
            if is_query:
                ax.set_title(lab, fontsize=8, color="blue")
            else:
                ax.set_title(lab, fontsize=8, color="green" if correct else "red")
    if title: fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def save_reliability_diagram(details, path, title="Reliability diagram"):
    if not details:
        return
    lo=np.array([d[0] for d in details]); hi=np.array([d[1] for d in details]); acc=np.array([d[2] for d in details]); conf=np.array([d[3] for d in details])
    fig,ax=plt.subplots(figsize=(6,5)); ax.plot([0,1],[0,1],"--",label="Perfect calibration"); ax.plot(conf,acc,marker="o",label="Model"); ax.set_xlabel("Mean confidence"); ax.set_ylabel("Accuracy"); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)
