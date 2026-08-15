"""Ground-truth tests for the named-axis baselines (D-036).

``construct_residual_pca`` is the strip test for CA-DVAE's alignment head, so its
geometry has to be exactly right or the comparison it enables is worthless:

1. the named block passes through untouched (alignment R² = 1 by construction);
2. the free block is the part of X the constructs do NOT explain — on train it is
   orthogonal to the construct block, which is what makes the two blocks
   complementary rather than redundant;
3. train-only fitting: nothing about the eval rows can move the fitted OLS
   coefficients or PCA basis (the CLAUDE.md leakage rule, proved by perturbation);
4. width behaves at the boundary where constructs already fill the latent budget.

Synthetic data with a known generative structure makes all four checkable exactly.
"""

from __future__ import annotations

import numpy as np

from cadvae.eval.representations import construct_residual_pca

N_TR, N_TE, K, P, D = 600, 200, 4, 12, 10      # train/eval rows, constructs, features, latent


def _data(seed: int = 0):
    """X = linear function of the constructs + independent structure + noise."""
    rng = np.random.default_rng(seed)
    C_tr = rng.standard_normal((N_TR, K))
    C_te = rng.standard_normal((N_TE, K))
    W = rng.standard_normal((K, P))
    extra_tr = rng.standard_normal((N_TR, P)) * 0.5
    extra_te = rng.standard_normal((N_TE, P)) * 0.5
    return C_tr, C_tr @ W + extra_tr, C_te, C_te @ W + extra_te


def test_named_block_passes_through_unchanged():
    C_tr, X_tr, C_te, X_te = _data()
    R_tr, R_te = construct_residual_pca(C_tr, X_tr, C_te, X_te, D)
    assert R_tr.shape == (N_TR, D) and R_te.shape == (N_TE, D)
    # the first K columns ARE the constructs — this is what makes R² = 1 by construction
    assert np.allclose(R_tr[:, :K], C_tr)
    assert np.allclose(R_te[:, :K], C_te)


def test_free_block_is_orthogonal_to_the_named_block_on_train():
    """OLS residuals are orthogonal to the design matrix, so the blocks are complementary."""
    C_tr, X_tr, C_te, X_te = _data()
    R_tr, _ = construct_residual_pca(C_tr, X_tr, C_te, X_te, D)
    free = R_tr[:, K:]
    assert free.shape[1] == D - K
    Cc = C_tr - C_tr.mean(0)
    Fc = free - free.mean(0)
    corr = (Cc.T @ Fc) / (N_TR * np.outer(Cc.std(0), np.maximum(Fc.std(0), 1e-12)))
    assert np.abs(corr).max() < 1e-8


def test_eval_rows_cannot_move_the_fit():
    """Perturbation proof of train-only fitting (CLAUDE.md leakage guard)."""
    C_tr, X_tr, C_te, X_te = _data()
    R_tr_a, _ = construct_residual_pca(C_tr, X_tr, C_te, X_te, D)
    rng = np.random.default_rng(99)
    R_tr_b, _ = construct_residual_pca(C_tr, X_tr,
                                       C_te + 50.0 * rng.standard_normal(C_te.shape),
                                       X_te * 7.0 + 3.0, D)
    assert np.allclose(R_tr_a, R_tr_b)


def test_deterministic():
    args = (*_data(), D)
    a_tr, a_te = construct_residual_pca(*args)
    b_tr, b_te = construct_residual_pca(*args)
    assert np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)


def test_free_block_has_a_floor_when_constructs_fill_the_budget():
    """Dataset B's case: 17 constructs against d=16 must still get a free block (D-064).

    Without the floor the free block would be empty and the control would collapse onto
    ``constructs`` alone, which is by far the weaker representation (0.4129 vs 0.5438 on
    Dataset A). That would flatter CA-DVAE for a reason unrelated to naming.
    """
    C_tr, X_tr, C_te, X_te = _data()
    for latent in (K, K - 2):                      # budget met exactly, and exceeded
        R_tr, R_te = construct_residual_pca(C_tr, X_tr, C_te, X_te, latent, min_free_dims=4)
        assert R_tr.shape == (N_TR, K + 4) and R_te.shape == (N_TE, K + 4)
        assert np.allclose(R_tr[:, :K], C_tr)      # named axes still pass through intact
        assert np.allclose(R_te[:, :K], C_te)


def test_floor_does_not_change_the_dataset_a_geometry():
    """Where the budget already allows more than the floor, the floor is inert."""
    C_tr, X_tr, C_te, X_te = _data()
    a, _ = construct_residual_pca(C_tr, X_tr, C_te, X_te, D, min_free_dims=4)
    b, _ = construct_residual_pca(C_tr, X_tr, C_te, X_te, D, min_free_dims=0)
    assert a.shape[1] == D and np.array_equal(a, b)


def test_free_block_recovers_structure_the_constructs_miss():
    """The residual PCs must carry the independent component, not noise alone."""
    rng = np.random.default_rng(3)
    C_tr = rng.standard_normal((N_TR, K))
    C_te = rng.standard_normal((N_TE, K))
    W = rng.standard_normal((K, P))
    hidden_tr = rng.standard_normal(N_TR)          # a factor NO construct explains
    hidden_te = rng.standard_normal(N_TE)
    load = rng.standard_normal(P)
    X_tr = C_tr @ W + np.outer(hidden_tr, load) + 0.01 * rng.standard_normal((N_TR, P))
    X_te = C_te @ W + np.outer(hidden_te, load) + 0.01 * rng.standard_normal((N_TE, P))

    R_tr, R_te = construct_residual_pca(C_tr, X_tr, C_te, X_te, D)
    # first residual PC should be (up to sign/scale) the hidden factor, on BOTH splits
    for free, hidden in ((R_tr[:, K], hidden_tr), (R_te[:, K], hidden_te)):
        assert abs(np.corrcoef(free, hidden)[0, 1]) > 0.99
