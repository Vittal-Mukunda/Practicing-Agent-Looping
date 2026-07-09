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
    """Fraction of true positives among the top ``k_frac`` highest-scored users."""
    n = max(1, int(round(len(scores) * k_frac)))
    top = np.argsort(scores)[::-1][:n]
    return float(y_true[top].mean())


def downstream_metrics(y_true: np.ndarray, scores: np.ndarray, k_frac: float = 0.10) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        f"prec_at_{int(k_frac * 100)}pct": precision_at_k(y_true, scores, k_frac),
        "base_rate": float(np.mean(y_true)),
    }


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
