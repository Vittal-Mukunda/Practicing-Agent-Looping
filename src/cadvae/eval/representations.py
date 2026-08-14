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

Deliberately conservative choice, in the baseline's favour: when the construct count
already meets or exceeds the latent budget ``d`` (Dataset B: 17 constructs, d = 16),
the representation is the full construct block and its width exceeds ``d`` rather
than being truncated. Truncating would drop named axes to satisfy a capacity budget
the baseline did not ask for; giving a baseline the benefit of the doubt is what
makes beating it meaningful.

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
                           latent_dim: int, seed: int = 0
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

    Returns ``(R_train, R_eval)``. Width is ``max(latent_dim, C.shape[1])``.
    """
    n_free = max(0, int(latent_dim) - C_train.shape[1])
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
