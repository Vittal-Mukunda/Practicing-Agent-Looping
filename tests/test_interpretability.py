"""Ground-truth tests for the interpretability metrics (MIG / SAP / axis alignment).

The estimators must (1) rank a perfectly disentangled code above an entangled
(rotated) code of the SAME information content, (2) respect known bounds,
(3) skip degenerate factors, and (4) be deterministic. Synthetic data with a known
generative structure makes all four checkable exactly.
"""

from __future__ import annotations

import numpy as np

from cadvae.eval.interpretability import (
    axis_alignment,
    interpretability_summary,
    mig,
    sap,
)

N, K, D = 4000, 3, 6      # samples, factors, latent dims


def _factors(seed=0):
    return np.random.default_rng(seed).standard_normal((N, K))


def _disentangled(V, seed=1):
    """Latent = factors on dims 0..K-1 (tiny noise) + pure-noise padding dims."""
    rng = np.random.default_rng(seed)
    Z = np.concatenate([V + 0.01 * rng.standard_normal(V.shape),
                        rng.standard_normal((N, D - K))], axis=1)
    return Z


def _entangled(V, seed=2):
    """Same information, mixed: latent = V @ Q for the K=3 Helmert rotation — a
    FIXED orthogonal matrix chosen because every row is spread over several axes
    (a random QR rotation can land near a permutation by luck, which is then
    *correctly* scored as partially disentangled — not what this fixture wants)."""
    rng = np.random.default_rng(seed)
    q = np.array([[1 / np.sqrt(3), 1 / np.sqrt(2), 1 / np.sqrt(6)],
                  [1 / np.sqrt(3), -1 / np.sqrt(2), 1 / np.sqrt(6)],
                  [1 / np.sqrt(3), 0.0, -2 / np.sqrt(6)]]).T
    Z = np.concatenate([V @ q, rng.standard_normal((N, D - K))], axis=1)
    return Z


def test_mig_orders_disentangled_above_entangled():
    V = _factors()
    m_dis = mig(_disentangled(V), V)["mig"]
    m_ent = mig(_entangled(V), V)["mig"]
    assert m_dis > 0.5                     # one axis carries ~all the MI
    assert m_ent < 0.15                    # Helmert mixing smears MI across axes
    assert m_dis > m_ent + 0.3


def test_sap_orders_disentangled_above_entangled():
    V = _factors()
    s_dis = sap(_disentangled(V), V)["sap"]
    s_ent = sap(_entangled(V), V)["sap"]
    assert s_dis > 0.8                     # R² gap ≈ 1 − 0
    assert s_ent < 0.3                     # Helmert rows: gaps 0, 0, ~0.5 → mean ~0.17
    assert s_dis > s_ent + 0.3


def test_bounds_and_shapes():
    V = _factors()
    m = mig(_disentangled(V), V)
    assert m["mi_matrix"].shape == (D, K)
    for v in m["per_factor"].values():
        assert 0.0 <= v <= 1.0
    s = sap(_disentangled(V), V)
    assert s["score_matrix"].shape == (D, K)
    assert 0.0 <= s["sap"] <= 1.0


def test_axis_alignment_points_at_the_right_dim():
    V = _factors()
    ax = axis_alignment(_disentangled(V), V, [f"c{j}" for j in range(K)])
    for j in range(K):
        assert ax[f"c{j}"]["best_dim"] == j            # factor j lives on dim j
        assert ax[f"c{j}"]["best_r2"] > 0.95


def test_degenerate_factor_is_nan_and_excluded():
    V = _factors()
    V[:, 1] = 3.14                                     # constant factor → H≈0
    m = mig(_disentangled(V), V, names=["a", "const", "b"])
    assert np.isnan(m["per_factor"]["const"])
    assert np.isfinite(m["mig"])                       # mean over remaining factors
    s = sap(_disentangled(V), V, names=["a", "const", "b"])
    assert np.isnan(s["per_factor"]["const"])


def test_deterministic_and_summary_serializable():
    import json
    V = _factors()
    Z = _disentangled(V)
    s1 = interpretability_summary(Z, V, [f"c{j}" for j in range(K)])
    s2 = interpretability_summary(Z, V, [f"c{j}" for j in range(K)])
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)
    assert set(s1) == {"mig", "sap", "mig_per_factor", "sap_per_factor", "axis_alignment"}
