"""Persona-assignment stability metrics (Phase 6, D-030).

CLAUDE.md evaluation protocol: "Stability: ARI/NMI of persona assignments across
>=5 seeds; bootstrap-resample persistence." Two estimators, both purely functions
of data given to them (no I/O here; the Phase-6 orchestrator supplies embeddings):

* **Cross-seed stability** (`pairwise_stability`): persona assignments for the SAME
  fixed user set under each seed's independently trained model, scored by mean
  pairwise Adjusted Rand Index (Hubert & Arabie 1985, chance-corrected) and
  Normalized Mutual Information (Strehl & Ghosh 2002). ARI/NMI are permutation
  invariant, so cluster-label switching across seeds does not depress the score —
  only genuine re-partitioning does.
* **Bootstrap persistence** (`bootstrap_persistence`): re-fit the clustering on
  bootstrap resamples of the embedding and compare full-population assignments to
  the reference fit (cf. von Luxburg 2010, "Clustering stability: an overview").
  High persistence = the persona partition is a property of the population, not of
  the particular sample.

All KMeans fits run single-threaded (D-028: bit-reproducible under one BLAS
reduction order). Model rebuilding consumes the state files saved by the Phase-5
sweep runner, so no retraining is needed.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from cadvae.models.cadvae import CADVAE


def rebuild_cadvae(state: dict) -> CADVAE:
    """Reconstruct a trained CA-DVAE (eval mode, CPU) from a sweep state file."""
    model = CADVAE(int(state["in_dim"]), int(state["latent_dim"]),
                   [int(h) for h in state["hidden_dims"]],
                   int(state["aligned_dims"]), int(state["n_constructs"]))
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def load_model_state(path) -> dict:
    """Load a sweep model file (tensors + plain-python metadata only)."""
    return torch.load(path, map_location="cpu", weights_only=True)


def kmeans_assign(Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fit K-means on Z and return hard assignments (single-threaded, D-028)."""
    if len(Z) < k:
        raise ValueError(f"cannot fit k={k} clusters on {len(Z)} samples")
    with threadpool_limits(limits=1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
    return km.labels_.astype(np.int64)


def pairwise_stability(assignments: dict[int, np.ndarray]) -> dict:
    """Mean pairwise ARI/NMI over all seed pairs (same items, aligned order)."""
    seeds = sorted(assignments)
    lengths = {len(assignments[s]) for s in seeds}
    if len(lengths) > 1:
        raise ValueError(f"assignment lengths differ across seeds: {sorted(lengths)}")
    pairs = list(combinations(seeds, 2))
    if not pairs:
        return {"n_seeds": len(seeds), "n_pairs": 0,
                "ari_mean": float("nan"), "ari_std": float("nan"),
                "nmi_mean": float("nan"), "nmi_std": float("nan"), "pairs": []}
    rows = []
    for a, b in pairs:
        rows.append({"seeds": [a, b],
                     "ari": float(adjusted_rand_score(assignments[a], assignments[b])),
                     "nmi": float(normalized_mutual_info_score(assignments[a], assignments[b]))})
    ari = np.array([r["ari"] for r in rows])
    nmi = np.array([r["nmi"] for r in rows])
    return {"n_seeds": len(seeds), "n_pairs": len(pairs),
            "ari_mean": float(ari.mean()),
            "ari_std": float(ari.std(ddof=1)) if len(pairs) > 1 else 0.0,
            "nmi_mean": float(nmi.mean()),
            "nmi_std": float(nmi.std(ddof=1)) if len(pairs) > 1 else 0.0,
            "pairs": rows}


def bootstrap_persistence(Z: np.ndarray, k: int, n_boot: int = 20, seed: int = 0) -> dict:
    """ARI between the reference partition of Z and partitions refit on bootstrap
    resamples (each refit predicts the FULL population)."""
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if len(Z) < k:
        raise ValueError(f"cannot fit k={k} clusters on {len(Z)} samples")
    with threadpool_limits(limits=1):
        ref = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
    ref_labels = ref.labels_
    rng = np.random.default_rng(seed)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(Z), len(Z), replace=True)
        with threadpool_limits(limits=1):
            km_b = KMeans(n_clusters=k, random_state=seed + 1 + b, n_init=10).fit(Z[idx])
        aris.append(float(adjusted_rand_score(ref_labels, km_b.predict(Z))))
    arr = np.array(aris)
    return {"n_boot": n_boot, "ari_mean": float(arr.mean()),
            "ari_std": float(arr.std(ddof=1)) if n_boot > 1 else 0.0,
            "ari_all": [round(a, 6) for a in aris]}


def universe_matrix(ps) -> tuple[np.ndarray, np.ndarray]:
    """All users of a prepared split (train+val+test), sorted by user id.

    Every seed partitions the SAME user universe, so after this sort the rows of
    different seeds' universes refer to the same users in the same order — the
    alignment cross-seed stability requires. Feature values still differ per seed
    (each seed's train-only scaler), which is correct: stability is a property of
    the full per-seed pipeline. The orchestrator asserts id equality across seeds.
    """
    uid = np.concatenate([ps.ids["train"], ps.ids["val"], ps.ids["test"]])
    X = np.vstack([ps.X_train, ps.X_val, ps.X_test])
    order = np.argsort(uid, kind="stable")
    return uid[order], X[order]


def stability_sample(n_universe: int, cap: int, sample_seed: int) -> np.ndarray:
    """Deterministic, seed-INDEPENDENT row sample of the sorted universe (D-030):
    the same users are scored under every model seed."""
    if n_universe <= cap:
        return np.arange(n_universe)
    rng = np.random.default_rng(sample_seed)
    return np.sort(rng.choice(n_universe, cap, replace=False))
