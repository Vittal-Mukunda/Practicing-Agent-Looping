"""Derived non-neural representations used as baselines (D-036).

Home for representations that are built from the *named construct targets* rather
than learned. They exist to answer the sharpest reviewer question the project faces:

    "If you want latent axes that carry the names of marketing constructs, why do
     you need a VAE at all? The constructs are deterministic functions of the input
     features — just use them as coordinates."

That question has teeth here. ``cadvae.data.constructs`` computes RFM, price
sensitivity and category affinity by arithmetic on the engineered feature table, so
the representation ``[C(x) | PCA(residual)]`` has per-axis alignment R² = 1 **by
construction**, costs no training, and has exactly the interpretability property
CA-DVAE claims. It is therefore the correct strip test for the alignment mechanism
(references/design-and-redteam.md): if CA-DVAE cannot beat it on downstream lift or
stability, the alignment machinery is decoration; if it can, the aligned block is
demonstrably more than a re-encoding of its own supervision targets.

Deliberately conservative choice, in the baseline's favour: named axes are never
truncated to fit the latent budget ``d``, and the free block is never dropped. The
free block is sized as ``max(d - n_constructs, min_free_dims)``, so the control always
carries at least as much free capacity as CA-DVAE itself does (D-064).

That floor matters. On Dataset A the budget arithmetic already gives 6 free components
and the rule changes nothing. On Dataset B there are 17 constructs against d = 16, so
``d - n_constructs`` is negative and a naive rule would hand the control zero free
components -- collapsing it onto ``constructs`` alone, which Section VI-E2 measured as
by far the weaker representation (0.4129 vs 0.5438 on A, d_z = 5.46). Testing CA-DVAE
against a control stripped of the one component this project proved load-bearing would
flatter the model for a reason unrelated to naming. The control is therefore wider than
``d`` on Dataset B (17 + 4 = 21 dimensions), and that is stated where the result is
reported.

Torch-free on purpose (numpy + scikit-learn only) so the geometry is unit-testable
without a GPU, a trained model, or the datasets.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


def _design(C: np.ndarray) -> np.ndarray:
    """Construct block with an intercept column (OLS design matrix)."""
    return np.column_stack([np.ones(len(C)), C])


def construct_residual_pca(C_train: np.ndarray, X_train: np.ndarray,
                           C_eval: np.ndarray, X_eval: np.ndarray,
                           latent_dim: int, seed: int = 0, min_free_dims: int = 4
                           ) -> tuple[np.ndarray, np.ndarray]:
    """``[C | PCA(X - X̂(C))]`` — named axes plus a free block, with no training.

    The free block is PCA of the part of the feature vector that the constructs do
    **not** linearly explain, so the two blocks carry complementary information — the
    same decomposition CA-DVAE is asked to learn, obtained in closed form.

    Leakage protocol (CLAUDE.md): the OLS coefficients and the PCA basis are fit on
    ``*_train`` only and applied to ``*_eval`` without refitting.

    Reproducibility (D-028): both fits run under ``threadpool_limits(1)`` —
    multithreaded-BLAS reduction order is not bit-stable, and this representation
    feeds downstream heads.

    ``min_free_dims`` is the floor on the free block, matching the model's own
    ``model.min_free_dims``; the control never gets less free capacity than CA-DVAE.

    Returns ``(R_train, R_eval)``. Width is
    ``C.shape[1] + max(latent_dim - C.shape[1], min_free_dims)``.
    """
    n_free = max(int(latent_dim) - C_train.shape[1], int(min_free_dims))
    if n_free == 0:
        return C_train, C_eval

    with threadpool_limits(limits=1):                            # (D-028)
        coef, *_ = np.linalg.lstsq(_design(C_train), X_train, rcond=None)
        resid_train = X_train - _design(C_train) @ coef
        n_comp = min(n_free, *resid_train.shape)
        pca = PCA(n_components=n_comp, svd_solver="full", random_state=seed)
        pca.fit(resid_train)
        free_train = pca.transform(resid_train)
        free_eval = pca.transform(X_eval - _design(C_eval) @ coef)

    return (np.hstack([C_train, free_train]), np.hstack([C_eval, free_eval]))
