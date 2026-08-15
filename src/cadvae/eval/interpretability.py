"""Interpretability / disentanglement metrics (Phase 5 sweep + Phase 6 analysis).

These score how well individual latent dimensions capture the *named* construct
factors — the x-axis of the paper's headline interpretability–performance
trade-off curve. Ground-truth "factors" = the Phase-2 construct targets
(standardized, train-only fit), which is the natural choice here: the claim under
test is precisely "latent axes align with named marketing constructs".

Implemented metrics (each with the standard citation for the methods section):

* **MIG — Mutual Information Gap.** Chen, Li, Grosse, Duvenaud, "Isolating Sources
  of Disentanglement in Variational Autoencoders", NeurIPS 2018 (sec. 4.1, eq. 6):
  for each factor v_k, the gap between the largest and second-largest mutual
  information I(z_j; v_k) over latent dims j, normalized by the factor entropy
  H(v_k); averaged over factors. MIG ∈ [0, 1]; high = each factor is captured by
  ONE axis rather than smeared over many. The gap term is deliberate: the authors
  state it penalizes the case where latents OTHER than the winner also encode the
  factor. Construct alignment distributes a construct across a block by design, so
  MIG scores against it — see D-063. Verified against the source 2026-08-15.
* **SAP — Separated Attribute Predictability.** Kumar, Sattigeri, Balakrishnan,
  "Variational Inference of Disentangled Latent Concepts from Unlabeled
  Observations", ICLR 2018 (sec. 3, continuous-attribute form): the score matrix
  S[j, k] is the R² of predicting factor k from single latent j, which the paper
  gives explicitly as the squared Pearson correlation
  (Cov(mu_i, y_k) / (sigma_mu_i * sigma_y_k))^2; SAP = mean over factors of
  (top-1 − top-2) of each factor's column. Verified against the source 2026-08-15.
* **Per-axis alignment score** (ours, D-026): for each construct, the best
  single-dim R² and which dim wins — the direct "is there a dim you can point at
  and name?" number used on persona cards.
* **Concept-leakage diagnostic** (D-037): axis purity (on-target minus off-target
  R² for the winning axis) and construct recoverability from the FREE block alone.
  A high alignment R² is necessary but NOT sufficient for the interpretability
  claim — see ``leakage_diagnostic`` for the two papers that establish why.

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


def _block_r2(Z_block: np.ndarray, v: np.ndarray) -> float:
    """R² of the least-squares linear prediction of factor ``v`` from a latent block.

    Multivariate (whole block, with intercept), unlike ``sap``'s single-dim scores.
    Returns 0.0 for an empty block and NaN for a constant factor.
    """
    if Z_block.shape[1] == 0:
        return 0.0
    ss_tot = float(((v - v.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return float("nan")
    A = np.column_stack([np.ones(len(Z_block)), Z_block])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    ss_res = float(((v - A @ coef) ** 2).sum())
    return float(1.0 - ss_res / ss_tot)


def leakage_diagnostic(Z: np.ndarray, V: np.ndarray, names: list[str],
                       aligned_dims: int) -> dict:
    """Concept-leakage / axis-purity check on the aligned–free latent partition.

    A high alignment R² on a named axis does NOT establish that the axis carries
    *only* that construct. Two independent groups document exactly this failure for
    concept-supervised models with an unsupervised side channel — which is precisely
    this architecture: Mahinpei, Clark, Lage, Doshi-Velez, Pan, "Promises and
    Pitfalls of Black-Box Concept Learning Models", 2021 (concept representations
    "encode information beyond the pre-defined concepts"), and Margeloiu, Ashman,
    Bhatt, Chen, Jamnik, Weller, "Do Concept Bottleneck Models Learn as Intended?",
    2021. The diagonal of the alignment matrix is therefore necessary evidence for
    the interpretability claim but not sufficient; this function reports the two
    quantities that complete it.

    Per construct:

    * ``on_target_r2`` / ``best_dim`` — the winning ALIGNED axis and its R² (the
      number the paper already reports).
    * ``off_target_r2`` — the largest R² that same axis has with a *different*
      construct. A named axis that also explains its neighbours is not separated.
    * ``purity`` = ``on_target_r2 - off_target_r2``. High = the axis earns its name.
    * ``free_block_r2`` — R² of predicting the construct from the FREE block alone.
      High = the construct is duplicated outside its named axes, so pointing at the
      named axis misdescribes where the information lives (the leakage signature).
    * ``aligned_block_r2`` — same, from the whole aligned block (reference).

    Summary scalars are means over constructs with finite values. ``leakage_ratio``
    = mean(free_block_r2) / mean(aligned_block_r2): ~0 means the naming is
    exclusive, ~1 means the free block knows as much as the named block does.

    Deterministic, no RNG. Z/V are the same matrices the other metrics take.
    """
    a = max(0, min(int(aligned_dims), Z.shape[1]))
    S = sap(Z, V, names)["score_matrix"]                 # (d, K) single-dim R²
    Z_aligned, Z_free = Z[:, :a], Z[:, a:]

    per: dict[str, dict] = {}
    for j, name in enumerate(names):
        if a == 0:
            best, on, off = -1, float("nan"), float("nan")
        else:
            best = int(np.argmax(S[:a, j]))
            on = float(S[best, j])
            others = np.delete(S[best, :], j)
            off = float(others.max()) if others.size else float("nan")
        per[name] = {
            "best_dim": best,
            "on_target_r2": on,
            "off_target_r2": off,
            "purity": float(on - off) if np.isfinite(on) and np.isfinite(off) else float("nan"),
            "aligned_block_r2": _block_r2(Z_aligned, V[:, j]),
            "free_block_r2": _block_r2(Z_free, V[:, j]),
        }

    def _mean(key: str) -> float:
        vals = [p[key] for p in per.values() if np.isfinite(p[key])]
        return float(np.mean(vals)) if vals else float("nan")

    free_mean, aligned_mean = _mean("free_block_r2"), _mean("aligned_block_r2")
    return {
        "per_construct": per,
        "axis_purity_mean": _mean("purity"),
        "free_block_r2_mean": free_mean,
        "aligned_block_r2_mean": aligned_mean,
        "leakage_ratio": (float(free_mean / aligned_mean)
                          if np.isfinite(free_mean) and np.isfinite(aligned_mean)
                          and abs(aligned_mean) > 1e-12 else float("nan")),
        "aligned_dims": a,
        "free_dims": int(Z.shape[1] - a),
    }


def interpretability_summary(Z: np.ndarray, V: np.ndarray, names: list[str],
                             n_bins: int = N_BINS_DEFAULT,
                             aligned_dims: int | None = None) -> dict:
    """Everything the sweep logs per run (JSON-serializable, matrices excluded).

    ``aligned_dims`` (optional) adds the concept-leakage block; omitted keeps the
    exact pre-existing key set, so already-recorded sweep artifacts stay comparable.
    """
    m = mig(Z, V, names, n_bins)
    s = sap(Z, V, names)
    ax = axis_alignment(Z, V, names)
    out = {"mig": m["mig"], "sap": s["sap"],
           "mig_per_factor": m["per_factor"], "sap_per_factor": s["per_factor"],
           "axis_alignment": ax}
    if aligned_dims is not None:
        lk = leakage_diagnostic(Z, V, names, aligned_dims)
        out["leakage"] = lk
        out["axis_purity_mean"] = lk["axis_purity_mean"]
        out["leakage_ratio"] = lk["leakage_ratio"]
    return out
