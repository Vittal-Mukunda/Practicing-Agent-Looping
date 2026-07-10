"""Interpretability / disentanglement metrics (Phase 5 sweep + Phase 6 analysis).

These score how well individual latent dimensions capture the *named* construct
factors — the x-axis of the paper's headline interpretability–performance
trade-off curve. Ground-truth "factors" = the Phase-2 construct targets
(standardized, train-only fit), which is the natural choice here: the claim under
test is precisely "latent axes align with named marketing constructs".

Implemented metrics (each with the standard citation for the methods section):

* **MIG — Mutual Information Gap.** Chen, Li, Grosse, Duvenaud, "Isolating Sources
  of Disentanglement in Variational Autoencoders", NeurIPS 2018 (eq. 6):
  for each factor v_k, the gap between the largest and second-largest mutual
  information I(z_j; v_k) over latent dims j, normalized by the factor entropy
  H(v_k); averaged over factors. MIG ∈ [0, 1]; high = each factor is captured by
  ONE axis rather than smeared over many.
* **SAP — Separated Attribute Predictability.** Kumar, Sattigeri, Balakrishnan,
  "Variational Inference of Disentangled Latent Concepts from Unlabeled
  Observations", ICLR 2018 (sec. 4.1, continuous-factor form): the score matrix
  S[j, k] is the R² of predicting factor k from single latent j (= squared Pearson
  correlation for the linear case); SAP = mean over factors of (top-1 − top-2) of
  each factor's column.
* **Per-axis alignment score** (ours, D-026): for each construct, the best
  single-dim R² and which dim wins — the direct "is there a dim you can point at
  and name?" number used on persona cards.

Estimator conventions follow Locatello et al., "Challenging Common Assumptions in
the Unsupervised Learning of Disentangled Representations", ICML 2019
(disentanglement_lib): mutual information on discretized variables with a fixed
bin count (default 20). Deviation, documented: their factors are categorical by
construction; our construct factors are continuous, so BOTH latents and factors
are discretized with **quantile** bins (equal-mass — robust to the heavy tails of
monetary/count features, unlike equal-width bins). Factors with (near-)zero
entropy after binning are skipped (reported as NaN, excluded from means).
Everything is deterministic — no RNG.
"""

from __future__ import annotations

import numpy as np

N_BINS_DEFAULT = 20


def _quantile_digitize(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize a 1-D array into ≤ n_bins equal-mass bins (deterministic).

    Duplicate quantile edges (heavily-tied variables, e.g. zero-inflated counts)
    are collapsed, so the effective bin count can be < n_bins — entropy is then
    computed on the bins that actually exist.
    """
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    return np.searchsorted(edges, x, side="right")


def _discrete_mi(zb: np.ndarray, vb: np.ndarray) -> float:
    """Mutual information (nats) between two already-discretized arrays."""
    n = len(zb)
    joint = np.zeros((int(zb.max()) + 1, int(vb.max()) + 1), dtype=np.float64)
    np.add.at(joint, (zb, vb), 1.0)
    joint /= n
    pz = joint.sum(axis=1, keepdims=True)
    pv = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pz @ pv)[nz])).sum())


def _entropy(vb: np.ndarray) -> float:
    p = np.bincount(vb) / len(vb)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mig(Z: np.ndarray, V: np.ndarray, names: list[str] | None = None,
        n_bins: int = N_BINS_DEFAULT) -> dict:
    """Mutual Information Gap of latents Z (n×d) w.r.t. factors V (n×K).

    Returns {"mig", "per_factor": {name: gap}, "mi_matrix": (d×K)}. Factors whose
    binned entropy is ~0 (constant) are NaN and excluded from the mean.
    """
    d, k = Z.shape[1], V.shape[1]
    names = names or [f"v{j}" for j in range(k)]
    Zb = np.column_stack([_quantile_digitize(Z[:, j], n_bins) for j in range(d)])
    mi = np.zeros((d, k))
    gaps: dict[str, float] = {}
    vals = []
    for j in range(k):
        vb = _quantile_digitize(V[:, j], n_bins)
        h = _entropy(vb)
        if h < 1e-12:
            gaps[names[j]] = float("nan")
            continue
        mi[:, j] = [_discrete_mi(Zb[:, i], vb) for i in range(d)]
        top2 = np.sort(mi[:, j])[-2:]
        gap = float((top2[1] - top2[0]) / h)
        gaps[names[j]] = gap
        vals.append(gap)
    return {"mig": (float(np.mean(vals)) if vals else float("nan")),
            "per_factor": gaps, "mi_matrix": mi}


def sap(Z: np.ndarray, V: np.ndarray, names: list[str] | None = None) -> dict:
    """SAP score (continuous-factor form): squared-Pearson score matrix, top-1 minus
    top-2 per factor, averaged. Returns {"sap", "per_factor", "score_matrix"}."""
    k = V.shape[1]
    names = names or [f"v{j}" for j in range(k)]
    Zs = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-12)
    Vs = (V - V.mean(0)) / np.maximum(V.std(0), 1e-12)
    S = (Zs.T @ Vs / len(Z)) ** 2                      # (d, K) squared correlations
    per: dict[str, float] = {}
    vals = []
    for j in range(k):
        if V[:, j].std() < 1e-12:
            per[names[j]] = float("nan")
            continue
        top2 = np.sort(S[:, j])[-2:]
        per[names[j]] = float(top2[1] - top2[0])
        vals.append(per[names[j]])
    return {"sap": (float(np.mean(vals)) if vals else float("nan")),
            "per_factor": per, "score_matrix": S}


def axis_alignment(Z: np.ndarray, V: np.ndarray, names: list[str]) -> dict:
    """Per-construct best single-axis R² + the winning dim (persona-card pointer)."""
    S = sap(Z, V, names)["score_matrix"]
    return {
        name: {"best_dim": int(np.argmax(S[:, j])), "best_r2": float(S[:, j].max())}
        for j, name in enumerate(names)
    }


def interpretability_summary(Z: np.ndarray, V: np.ndarray, names: list[str],
                             n_bins: int = N_BINS_DEFAULT) -> dict:
    """Everything the sweep logs per run (JSON-serializable, matrices excluded)."""
    m = mig(Z, V, names, n_bins)
    s = sap(Z, V, names)
    ax = axis_alignment(Z, V, names)
    return {"mig": m["mig"], "sap": s["sap"],
            "mig_per_factor": m["per_factor"], "sap_per_factor": s["per_factor"],
            "axis_alignment": ax}
