"""Smoke tests for the CA-DVAE (Phase 4): the model + loss + both ablation config
points train end-to-end on tiny synthetic data (CPU, few epochs), produce finite,
well-shaped outputs, and are deterministic given a seed.

These check the *machinery*, not the science (that is the sanity run + Phase 5).
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from cadvae.models.cadvae import (
    CADVAE,
    align_predict,
    cadvae_loss,
    encode,
    resolve_aligned_dims,
    train_cadvae,
)

N, D, C = 160, 6, 5          # samples, features, constructs


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D)).astype(np.float64)
    # constructs partly linear in X so alignment has real signal to fit
    Cm = (X[:, :3] @ rng.standard_normal((3, C)) + 0.1 * rng.standard_normal((N, C)))
    return X, Cm.astype(np.float64)


def _cfg(latent=8, aligned=None, beta=4.0, lam=1.0, epochs=3):
    return OmegaConf.create({
        "latent_dim": latent, "hidden_dims": [16], "aligned_dims": aligned,
        "min_free_dims": 2, "beta": beta, "lambda_align": lam,
        "aligned_kl_weight": 1.0, "lr": 1e-3, "batch_size": 64, "max_epochs": epochs,
    })


def test_resolve_aligned_dims_rules():
    # auto: min(n_constructs, latent - min_free) = min(5, 8-2) = 5
    assert resolve_aligned_dims(_cfg(latent=8), n_constructs=5) == 5
    # capped by (latent - min_free): min(20, 8-2) = 6
    assert resolve_aligned_dims(_cfg(latent=8), n_constructs=20) == 6
    # explicit value honored
    assert resolve_aligned_dims(_cfg(latent=8, aligned=3), n_constructs=5) == 3
    # lambda_align == 0 forces a pure β-VAE (no aligned block) regardless of constructs
    assert resolve_aligned_dims(_cfg(latent=8, aligned=5, lam=0.0), n_constructs=5) == 0


def test_cadvae_full_trains_and_shapes():
    X, Cm = _data()
    model, hist = train_cadvae(X, Cm, _cfg(latent=8), seed=0, device="cpu", n_constructs=C)
    assert len(hist) == 3
    for e in hist:
        for k in ("total", "recon", "kl_free", "kl_aligned", "align"):
            assert np.isfinite(e[k])
    # KL terms are non-negative by construction (Gaussian KL vs N(0,1))
    assert all(e["kl_free"] >= -1e-6 and e["kl_aligned"] >= -1e-6 for e in hist)
    emb = encode(model, X, device="cpu")
    assert emb.shape == (N, 8)
    pred = align_predict(model, X, device="cpu")
    assert pred is not None and pred.shape == (N, C)


def test_betavae_ablation_has_no_alignment():
    """lambda_align=0 → aligned_dims 0, no align head, β weights the whole KL."""
    X, _ = _data()
    model, hist = train_cadvae(X, None, _cfg(lam=0.0), seed=0, device="cpu", n_constructs=C)
    assert model.align_head is None
    assert model.aligned_dims == 0
    assert align_predict(model, X, device="cpu") is None
    # all KL sits in kl_free (aligned block empty), alignment term is exactly 0
    assert all(e["kl_aligned"] == 0.0 and e["align"] == 0.0 for e in hist)


def test_vae_plus_align_ablation_trains():
    """beta=1 → VAE + alignment, no extra disentanglement pressure."""
    X, Cm = _data()
    model, hist = train_cadvae(X, Cm, _cfg(beta=1.0), seed=0, device="cpu", n_constructs=C)
    assert model.align_head is not None
    assert all(np.isfinite(e["total"]) for e in hist)


def test_cadvae_deterministic():
    X, Cm = _data()
    m1, _ = train_cadvae(X, Cm, _cfg(), seed=3, device="cpu", n_constructs=C)
    m2, _ = train_cadvae(X, Cm, _cfg(), seed=3, device="cpu", n_constructs=C)
    assert np.allclose(encode(m1, X, device="cpu"), encode(m2, X, device="cpu"))


def test_loss_terms_reduce_to_expected_scalars():
    """cadvae_loss returns finite scalars and honors the aligned/free KL split."""
    X, Cm = _data()
    model = CADVAE(D, 8, [16], aligned_dims=5, n_constructs=C)
    out = model(torch.as_tensor(X, dtype=torch.float32))
    losses = cadvae_loss(out, torch.as_tensor(X, dtype=torch.float32),
                         torch.as_tensor(Cm, dtype=torch.float32),
                         aligned_dims=5, beta=4.0, lambda_align=1.0, aligned_kl_weight=1.0)
    assert all(torch.isfinite(v) for v in losses.values())
    assert losses["kl_aligned"].item() >= 0.0 and losses["kl_free"].item() >= 0.0
