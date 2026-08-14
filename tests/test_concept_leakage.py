"""Ground-truth tests for the concept-leakage diagnostic (D-037).

The interpretability claim rests on "this axis IS recency". Two independent groups
(Mahinpei et al. 2021; Margeloiu et al. 2021) show that concept-supervised models
with an unsupervised side channel can post a high on-target score while the concept
information actually lives elsewhere — so the diagnostic is only worth reporting if
it can tell a clean latent space from a leaky one. These tests construct both cases
with known ground truth and require the metric to separate them.
"""

from __future__ import annotations

import numpy as np

from cadvae.eval.interpretability import leakage_diagnostic

N, K = 4000, 3
NAMES = ["rfm_R", "rfm_F", "rfm_M"]


def _factors(seed: int = 0):
    return np.random.default_rng(seed).standard_normal((N, K))


def test_clean_partition_scores_pure_and_unleaked():
    """Aligned dims carry one construct each; free dims are pure noise."""
    V = _factors()
    rng = np.random.default_rng(1)
    Z = np.hstack([V + 0.01 * rng.standard_normal(V.shape),      # 3 aligned
                   rng.standard_normal((N, 3))])                 # 3 free (noise)
    out = leakage_diagnostic(Z, V, NAMES, aligned_dims=K)

    assert out["aligned_dims"] == K and out["free_dims"] == 3
    assert out["axis_purity_mean"] > 0.9                 # on-target >> off-target
    assert out["free_block_r2_mean"] < 0.05              # free block knows nothing
    assert out["aligned_block_r2_mean"] > 0.95
    assert out["leakage_ratio"] < 0.05
    for j, name in enumerate(NAMES):
        assert out["per_construct"][name]["best_dim"] == j


def test_leaky_free_block_is_detected():
    """SAME on-target alignment, but the free block also carries the constructs.

    This is the failure the diagnostic exists to catch: axis_alignment alone cannot
    distinguish this case from the clean one, because the diagonal is identical.
    """
    V = _factors()
    rng = np.random.default_rng(1)
    aligned = V + 0.01 * rng.standard_normal(V.shape)
    Z_clean = np.hstack([aligned, rng.standard_normal((N, 3))])
    Z_leaky = np.hstack([aligned, V + 0.01 * rng.standard_normal(V.shape)])

    clean = leakage_diagnostic(Z_clean, V, NAMES, aligned_dims=K)
    leaky = leakage_diagnostic(Z_leaky, V, NAMES, aligned_dims=K)

    # the on-target evidence the paper currently reports is indistinguishable ...
    for name in NAMES:
        assert abs(clean["per_construct"][name]["on_target_r2"]
                   - leaky["per_construct"][name]["on_target_r2"]) < 0.02
    # ... but the diagnostic separates them decisively
    assert leaky["free_block_r2_mean"] > 0.95
    assert leaky["leakage_ratio"] > 0.95
    assert leaky["free_block_r2_mean"] > 10 * clean["free_block_r2_mean"]


def test_entangled_axis_loses_purity():
    """An aligned dim that explains two constructs at once must not read as 'named'."""
    V = _factors()
    rng = np.random.default_rng(2)
    mixed = (V[:, 0] + V[:, 1]) / np.sqrt(2.0)
    Z = np.hstack([mixed[:, None], mixed[:, None], V[:, 2:3],
                   rng.standard_normal((N, 3))])
    out = leakage_diagnostic(Z, V, NAMES, aligned_dims=K)

    assert out["per_construct"]["rfm_R"]["purity"] < 0.1     # explains F just as well
    assert out["per_construct"]["rfm_M"]["purity"] > 0.9     # genuinely its own axis
    assert out["axis_purity_mean"] < out["per_construct"]["rfm_M"]["purity"]


def test_no_aligned_dims_is_nan_not_a_crash():
    """lambda = 0 ablation (pure beta-VAE): there are no named axes to score."""
    V = _factors()
    Z = np.random.default_rng(3).standard_normal((N, 6))
    out = leakage_diagnostic(Z, V, NAMES, aligned_dims=0)
    assert out["aligned_dims"] == 0 and out["free_dims"] == 6
    assert np.isnan(out["axis_purity_mean"])
    assert out["aligned_block_r2_mean"] == 0.0              # empty block explains nothing


def test_deterministic_and_summary_wiring():
    from cadvae.eval.interpretability import interpretability_summary

    V = _factors()
    rng = np.random.default_rng(1)
    Z = np.hstack([V + 0.01 * rng.standard_normal(V.shape), rng.standard_normal((N, 3))])

    a = leakage_diagnostic(Z, V, NAMES, aligned_dims=K)
    b = leakage_diagnostic(Z, V, NAMES, aligned_dims=K)
    assert a == b

    # opt-in: absent by default so already-recorded sweep artifacts stay comparable
    assert "leakage" not in interpretability_summary(Z, V, NAMES)
    s = interpretability_summary(Z, V, NAMES, aligned_dims=K)
    assert s["leakage"]["axis_purity_mean"] == a["axis_purity_mean"]
    assert s["axis_purity_mean"] == a["axis_purity_mean"]
