"""Evaluation metrics (Phase 3).

Downstream (frozen-representation prediction): ROC-AUC, PR-AUC (primary — both
labels are imbalanced), and precision@k (top-k targeting quality).
Clustering: silhouette / Davies–Bouldin / Calinski–Harabasz (silhouette on a capped
sample for tractability on multi-million-row Dataset B).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_frac: float = 0.10) -> float:
    """Expected precision among the top ``k_frac`` highest-scored users, tie-aware (D-025).

    Tree heads and cluster one-hots emit piecewise-constant scores, so exact ties at
    the top-k boundary are common. Naive argsort-based top-k then depends on the row
    order on disk (not reproducible, and silently arbitrary). Instead the ties at the
    boundary score contribute their EXPECTED positive rate for the remaining slots —
    the average over all tie-breaking orders: deterministic and row-order invariant.
    Equal to the plain top-k precision whenever there are no boundary ties."""
    n = max(1, int(round(len(scores) * k_frac)))
    order = np.argsort(scores, kind="stable")[::-1]
    thresh = scores[order[n - 1]]
    above = scores > thresh
    tied = scores == thresh
    n_above = int(above.sum())
    hits = float(y_true[above].sum())
    slots_left = n - n_above                      # >= 1 by construction of thresh
    hits += slots_left * float(y_true[tied].mean())
    return hits / n


def downstream_metrics(y_true: np.ndarray, scores: np.ndarray, k_frac: float = 0.10) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        f"prec_at_{int(k_frac * 100)}pct": precision_at_k(y_true, scores, k_frac),
        "base_rate": float(np.mean(y_true)),
    }


def multiclass_metrics(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray,
                       top_ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """Multiclass downstream metrics (next-category task, D-016/D-023).

    * ``acc_at_k``: is the true class among the k highest-probability classes?
      (CLAUDE.md "top-k" for next-purchase-category.)
    * ``macro_ovr_auc``: mean one-vs-rest ROC-AUC over the classes present in
      ``y_true`` — computed per class from that class's own probability column, so no
      renormalization is needed and absent classes are simply skipped.
    """
    class_pos = {c: i for i, c in enumerate(classes)}
    # rank of the true class's probability within each row (0 = highest)
    order = np.argsort(proba, axis=1, kind="stable")[:, ::-1]
    true_col = np.array([class_pos.get(c, -1) for c in y_true])
    known = true_col >= 0                       # true class seen in train
    ranks = np.full(len(y_true), proba.shape[1], dtype=np.int64)  # unseen → never hit
    if known.any():
        hit_pos = order[known] == true_col[known, None]
        ranks[known] = hit_pos.argmax(axis=1)
    out: dict = {"base_rate": float(np.bincount(true_col[known]).max() / len(y_true))
                 if known.any() else float("nan"),
                 "n_classes": int(len(classes))}
    for k in top_ks:
        out[f"acc_at_{k}"] = float((ranks < k).mean())
    aucs = []
    for c, i in class_pos.items():
        pos = (y_true == c)
        if 0 < pos.sum() < len(y_true):
            aucs.append(roc_auc_score(pos, proba[:, i]))
    out["macro_ovr_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    return out


def clustering_metrics(embedding: np.ndarray, labels: np.ndarray,
                       sample_cap: int = 20000, seed: int = 0) -> dict:
    """Silhouette/DB/CH on the representation space. Silhouette (O(n^2)) uses a
    capped random sample; DB/CH are cheap and use all points."""
    n_labels = len(np.unique(labels))
    if n_labels < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan"), "n_clusters_found": n_labels}
    rng = np.random.default_rng(seed)
    if len(embedding) > sample_cap:
        idx = rng.choice(len(embedding), sample_cap, replace=False)
        emb_s, lab_s = embedding[idx], labels[idx]
    else:
        emb_s, lab_s = embedding, labels
    sil = (silhouette_score(emb_s, lab_s) if len(np.unique(lab_s)) >= 2 else float("nan"))
    return {
        "silhouette": float(sil),
        "davies_bouldin": float(davies_bouldin_score(embedding, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(embedding, labels)),
        "n_clusters_found": int(n_labels),
    }
