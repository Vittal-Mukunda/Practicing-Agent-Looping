"""Frozen-representation downstream protocol (Phase 3, D-019).

For one (dataset, seed): learn each baseline's representation on **train only**,
freeze it, train two simple heads (L2-logistic + HistGBT) on the frozen train
representation, and evaluate on the **held-out test** set. Clustering baselines also
get clustering-quality metrics on the representation space.

Nothing here touches the downstream label during representation learning — the label
enters only when the heads are fit.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from cadvae.eval.metrics import clustering_metrics, downstream_metrics
from cadvae.models.ae import encode, train_autoencoder
from cadvae.models.dec import dec_assign, train_dec


def _onehot(labels: np.ndarray, k: int) -> np.ndarray:
    return np.eye(k, dtype=np.float64)[labels]


def _fit_eval_heads(R_tr, y_tr, R_te, y_te, seed: int) -> dict:
    """Two frozen-representation heads → test metrics keyed by head."""
    out = {}
    logreg = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    logreg.fit(R_tr, y_tr)
    out["logreg"] = downstream_metrics(y_te, logreg.predict_proba(R_te)[:, 1])
    gbt = HistGradientBoostingClassifier(random_state=seed)
    gbt.fit(R_tr, y_tr)
    out["gbt"] = downstream_metrics(y_te, gbt.predict_proba(R_te)[:, 1])
    return out


def _subsample(n: int, cfg, seed: int) -> np.ndarray:
    cap = cfg.eval.get("train_subsample")
    idx = np.arange(n)
    if cap is not None and n > int(cap):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, int(cap), replace=False))
    return idx


def evaluate_baselines(ps, ks, cfg, seed: int, device: str = "cuda") -> list[dict]:
    """Run every Phase-3 baseline for one seed → list of metric records."""
    k = int(cfg.eval.n_clusters)
    tr = _subsample(len(ps.y_train), cfg, seed)               # train subsample (B tractability)
    Xtr, ytr = ps.X_train[tr], ps.y_train[tr]
    Xte, yte = ps.X_test, ps.y_test
    records: list[dict] = []

    def emit(method, heads, clustering=None):
        for head, m in heads.items():
            rec = {"method": method, "head": head, "seed": seed, **m}
            if clustering:
                rec.update({f"clu_{ck}": cv for ck, cv in clustering.items()})
            records.append(rec)

    # --- RFM (incumbent floor, continuous) + RFM+K-means (segment) ---
    rfm_tr, rfm_te = ks.C_train[tr, :3], ks.C_test[:, :3]
    emit("rfm", _fit_eval_heads(rfm_tr, ytr, rfm_te, yte, seed))
    km_rfm = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(rfm_tr)
    emit("rfm_kmeans",
         _fit_eval_heads(_onehot(km_rfm.predict(rfm_tr), k), ytr,
                         _onehot(km_rfm.predict(rfm_te), k), yte, seed),
         clustering_metrics(rfm_te, km_rfm.predict(rfm_te), seed=seed))

    # --- AE embedding (the hard baseline) + AE-based clustering ---
    ae = train_autoencoder(Xtr, cfg.model, seed, device=device)
    # float64 for numerically-stable sklearn clustering (torch encode returns float32)
    Etr = encode(ae, Xtr, device=device).astype(np.float64)
    Ete = encode(ae, Xte, device=device).astype(np.float64)
    emit("ae", _fit_eval_heads(Etr, ytr, Ete, yte, seed))

    km_ae = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Etr)
    emit("ae_kmeans",
         _fit_eval_heads(_onehot(km_ae.predict(Etr), k), ytr,
                         _onehot(km_ae.predict(Ete), k), yte, seed),
         clustering_metrics(Ete, km_ae.predict(Ete), seed=seed))

    # diagonal covariance + reg_covar=1e-3: robust to near-collapsed AE latent dims
    # (full covariance is singular when a dim degenerates); standard for GMM-on-embeddings
    gmm = GaussianMixture(n_components=k, random_state=seed, reg_covar=1e-3,
                          covariance_type="diag").fit(Etr)
    emit("gmm_ae",
         _fit_eval_heads(gmm.predict_proba(Etr), ytr, gmm.predict_proba(Ete), yte, seed),
         clustering_metrics(Ete, gmm.predict(Ete), seed=seed))

    # --- DEC ---
    dec = train_dec(ae, Xtr, cfg.model, k, seed, device=device)
    q_tr, q_te = dec_assign(dec, Xtr), dec_assign(dec, Xte)
    emit("dec",
         _fit_eval_heads(q_tr, ytr, q_te, yte, seed),
         clustering_metrics(Ete, q_te.argmax(1), seed=seed))

    return records
