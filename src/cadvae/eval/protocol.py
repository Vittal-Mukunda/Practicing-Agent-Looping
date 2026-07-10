"""Frozen-representation downstream protocol (Phase 3, D-019; extended D-024/028).

For one (dataset, seed): learn each baseline's representation on **train only**,
freeze it, train simple heads (L2-logistic + HistGBT) on the frozen train
representation, and evaluate on the **held-out test** set. Clustering baselines also
get clustering-quality metrics on the representation space.

Methods (``eval.methods`` selects a subset; default = all):

* ``raw``        — the full standardized feature vector, no representation learning.
                   A REFERENCE CEILING, not a persona method (D-024): it answers the
                   reviewer question "do you even need a representation?" and is
                   excluded from bar/best-baseline selection.
* ``pca``        — classical linear compression at the same capacity (d = latent_dim).
* ``rfm`` / ``rfm_kmeans`` — incumbent floor (continuous / segmented).
* ``ae`` / ``ae_kmeans`` / ``gmm_ae`` — the hard neural baseline + clusterings.
* ``dec``        — Deep Embedded Clustering.

Multi-task (D-016/D-023): ``tasks`` carries extra downstream labels (Dataset B:
churned, next_category) evaluated on the SAME frozen representation — the
representation is learned once per method; only the heads are refit per task.
Records carry a ``task`` field ("primary" = the dataset's main label).

Reproducibility (D-028): every sklearn *fit* that feeds a representation
(KMeans/GMM/PCA) runs under ``threadpool_limits(1)`` — multithreaded-BLAS reduction
order makes fits non-bit-reproducible (~1e-7 wobble measured on this env), which is
absorbed by discrete outputs but leaks through continuous ones (GMM posteriors) and
explodes chaotically in DEC. Heads stay multithreaded (bit-identical across repeated
full runs, monitored by the Phase-7 identity check).

Nothing here touches any downstream label during representation learning — labels
enter only when heads are fit.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from cadvae.eval.metrics import clustering_metrics, downstream_metrics, multiclass_metrics
from cadvae.models.ae import encode, train_autoencoder
from cadvae.models.dec import dec_assign, train_dec

ALL_METHODS = ("raw", "pca", "rfm", "rfm_kmeans", "ae", "ae_kmeans", "gmm_ae", "dec")


def _onehot(labels: np.ndarray, k: int) -> np.ndarray:
    return np.eye(k, dtype=np.float64)[labels]


def _fit_eval_heads(R_tr, y_tr, R_te, y_te, seed: int) -> dict:
    """Two frozen-representation heads → test metrics keyed by head (binary task)."""
    out = {}
    logreg = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    logreg.fit(R_tr, y_tr)
    out["logreg"] = downstream_metrics(y_te, logreg.predict_proba(R_te)[:, 1])
    gbt = HistGradientBoostingClassifier(random_state=seed)
    gbt.fit(R_tr, y_tr)
    out["gbt"] = downstream_metrics(y_te, gbt.predict_proba(R_te)[:, 1])
    return out


def _fit_eval_heads_multiclass(R_tr, y_tr, R_te, y_te, seed: int,
                               top_ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """Same two heads for a multiclass task → top-k accuracy + macro-OVR AUC.

    ``early_stopping=False`` on the GBT head: the auto rule enables it above 10k
    samples, and its internal STRATIFIED validation split crashes when a rare class
    has a single member — real-tail category labels do (observed on Dataset B,
    unsubsampled). At the D-020 subsample the multiclass fit is below the threshold,
    so this changes nothing for the canonical protocol run."""
    out = {}
    logreg = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    logreg.fit(R_tr, y_tr)
    out["logreg"] = multiclass_metrics(y_te, logreg.predict_proba(R_te),
                                       logreg.classes_, top_ks)
    gbt = HistGradientBoostingClassifier(random_state=seed, early_stopping=False)
    gbt.fit(R_tr, y_tr)
    out["gbt"] = multiclass_metrics(y_te, gbt.predict_proba(R_te), gbt.classes_, top_ks)
    return out


def _subsample(n: int, cfg, seed: int) -> np.ndarray:
    cap = cfg.eval.get("train_subsample")
    idx = np.arange(n)
    if cap is not None and n > int(cap):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, int(cap), replace=False))
    return idx

def evaluate_baselines(ps, ks, cfg, seed: int, device: str = "cuda",
                       tasks: dict | None = None) -> list[dict]:
    """Run the requested Phase-3 baselines for one seed → list of metric records.

    ``tasks``: optional extra downstream tasks — {name: {"type": "binary"|"multiclass",
    "train": (y, mask), "test": (y, mask), "top_ks": (...)}} with y/mask aligned to the
    FULL train/test rows (mask selects the rows where the task is defined). The primary
    task (ps.y_*) is always evaluated and recorded as task="primary".
    """
    k = int(cfg.eval.n_clusters)
    methods = set(cfg.eval.get("methods") or ALL_METHODS)
    unknown = methods - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"unknown eval.methods: {sorted(unknown)}")
    tr = _subsample(len(ps.y_train), cfg, seed)               # train subsample (B tractability)
    Xtr, ytr = ps.X_train[tr], ps.y_train[tr]
    Xte, yte = ps.X_test, ps.y_test
    records: list[dict] = []

    def emit(method, R_tr, R_te, clustering=None):
        """Fit heads on the frozen rep for the primary + every extra task."""
        if method not in methods:
            return
        for head, m in _fit_eval_heads(R_tr, ytr, R_te, yte, seed).items():
            rec = {"method": method, "head": head, "seed": seed, "task": "primary", **m}
            if clustering:
                rec.update({f"clu_{ck}": cv for ck, cv in clustering.items()})
            records.append(rec)
        for tname, t in (tasks or {}).items():
            ytr_t, mtr = t["train"]
            yte_t, mte = t["test"]
            mtr = mtr[tr]                                     # align mask to the subsample
            ytr_t = ytr_t[tr]
            fit = (_fit_eval_heads_multiclass if t["type"] == "multiclass"
                   else _fit_eval_heads)
            kwargs = {"top_ks": tuple(t.get("top_ks", (1, 3, 5)))} \
                if t["type"] == "multiclass" else {}
            heads = fit(R_tr[mtr], ytr_t[mtr], R_te[mte], yte_t[mte], seed, **kwargs)
            for head, m in heads.items():
                records.append({"method": method, "head": head, "seed": seed,
                                "task": tname, **m})

    # --- raw standardized features: reference ceiling, excluded from the bar (D-024)
    emit("raw", Xtr, Xte)

    # --- PCA at matched capacity (classical linear baseline, D-024) ---
    if "pca" in methods:
        d_lat = min(int(cfg.model.latent_dim), Xtr.shape[1])
        with threadpool_limits(limits=1):                     # bit-reproducible fit (D-028)
            pca = PCA(n_components=d_lat, svd_solver="full", random_state=seed).fit(Xtr)
        emit("pca", pca.transform(Xtr), pca.transform(Xte))

    # --- RFM (incumbent floor, continuous) + RFM+K-means (segment) ---
    rfm_tr, rfm_te = ks.C_train[tr, :3], ks.C_test[:, :3]
    emit("rfm", rfm_tr, rfm_te)
    if "rfm_kmeans" in methods:
        with threadpool_limits(limits=1):                     # (D-028)
            km_rfm = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(rfm_tr)
        emit("rfm_kmeans",
             _onehot(km_rfm.predict(rfm_tr), k), _onehot(km_rfm.predict(rfm_te), k),
             clustering_metrics(rfm_te, km_rfm.predict(rfm_te), seed=seed))

    # --- AE embedding (the hard baseline) + AE-based clustering ---
    if methods & {"ae", "ae_kmeans", "gmm_ae", "dec"}:
        ae = train_autoencoder(Xtr, cfg.model, seed, device=device)
        # float64 for numerically-stable sklearn clustering (torch encode → float32)
        Etr = encode(ae, Xtr, device=device).astype(np.float64)
        Ete = encode(ae, Xte, device=device).astype(np.float64)
        emit("ae", Etr, Ete)

        if "ae_kmeans" in methods:
            with threadpool_limits(limits=1):                 # (D-028)
                km_ae = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Etr)
            emit("ae_kmeans",
                 _onehot(km_ae.predict(Etr), k), _onehot(km_ae.predict(Ete), k),
                 clustering_metrics(Ete, km_ae.predict(Ete), seed=seed))

        if "gmm_ae" in methods:
            # diagonal covariance + reg_covar=1e-3: robust to near-collapsed AE latent
            # dims (full covariance goes singular); standard for GMM-on-embeddings
            with threadpool_limits(limits=1):                 # (D-028)
                gmm = GaussianMixture(n_components=k, random_state=seed, reg_covar=1e-3,
                                      covariance_type="diag").fit(Etr)
            emit("gmm_ae", gmm.predict_proba(Etr), gmm.predict_proba(Ete),
                 clustering_metrics(Ete, gmm.predict(Ete), seed=seed))

        # --- DEC (fine-tunes the shared AE's encoder IN PLACE — must run last in
        #     the AE family so no other method sees the mutated weights) ---
        if "dec" in methods:
            dec = train_dec(ae, Xtr, cfg.model, k, seed, device=device)
            q_tr, q_te = dec_assign(dec, Xtr), dec_assign(dec, Xte)
            emit("dec", q_tr, q_te, clustering_metrics(Ete, q_te.argmax(1), seed=seed))

    return records
