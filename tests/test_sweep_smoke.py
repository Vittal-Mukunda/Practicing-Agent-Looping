"""Smoke tests for the Phase-5 sweep runner (tiny synthetic data, CPU)."""

from __future__ import annotations

import json

import numpy as np
from omegaconf import OmegaConf

from cadvae.data.constructs import ConstructSet
from cadvae.data.personality import PreparedSplit
from cadvae.eval.run_cadvae_sweep import _run_tag, run_point


def _tiny(seed=0, n=240, d=8, k=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(np.int64)
    tr, te = slice(0, n - 60), slice(n - 60, n)
    ps = PreparedSplit(
        X_train=X[tr], X_val=X[te], X_test=X[te],
        y_train=y[tr], y_val=y[te], y_test=y[te],
        feature_names=[f"f{i}" for i in range(d)],
        ids={"train": np.arange(n - 60), "val": np.arange(60), "test": np.arange(60)},
    )
    C = X[:, :k] + 0.1 * rng.standard_normal((n, k))
    ks = ConstructSet(C_train=C[tr], C_val=C[te], C_test=C[te],
                      names=[f"c{i}" for i in range(k)],
                      raw={"train": C[tr], "val": C[te], "test": C[te]})
    return ps, ks


def _cfg():
    return OmegaConf.create({
        "device": "cpu",
        "data": {"name": "tiny"},
        "model": {"name": "cadvae", "latent_dim": 6, "hidden_dims": [16], "lr": 1e-3,
                  "batch_size": 64, "max_epochs": 2, "aligned_dims": None,
                  "min_free_dims": 2, "beta": 1.0, "lambda_align": 1.0,
                  "aligned_kl_weight": 1.0},
        "eval": {"train_subsample": None, "seeds": [0], "n_clusters": 3,
                 "k_frac": 0.10, "primary_metric": "pr_auc"},
        "sweep": {"betas": [1.0], "lambdas": [0.0, 1.0]},
    })


def _tiny_tasks(ps, seed=0):
    rng = np.random.default_rng(seed + 9)
    ntr, nte = len(ps.y_train), len(ps.y_test)
    return {"churned": {"type": "binary",
                        "train": ((rng.random(ntr) < 0.5).astype(np.int64),
                                  np.ones(ntr, bool)),
                        "test": ((rng.random(nte) < 0.5).astype(np.int64),
                                 np.ones(nte, bool))}}


def test_run_point_full_model_record_complete_and_serializable():
    ps, ks = _tiny()
    rec, state = run_point(ps, ks, _tiny_tasks(ps), _cfg(), seed=0, beta=0.5, lam=1.0,
                           device="cpu")
    json.dumps(rec)                                       # must be JSON-serializable
    assert rec["aligned_dims"] == 3                       # min(3 constructs, 6-2 free)
    assert {"primary", "churned"} <= set(rec["downstream"])
    assert 0.0 <= rec["downstream"]["primary"]["gbt"]["pr_auc"] <= 1.0
    assert np.isfinite(rec["interpretability"]["mig"])
    assert "r2_mean" in rec["alignment"]
    assert state["aligned_dims"] == 3 and "state_dict" in state


def test_run_point_beta_vae_ablation_lambda_zero():
    ps, ks = _tiny()
    rec, state = run_point(ps, ks, {}, _cfg(), seed=0, beta=1.0, lam=0.0, device="cpu")
    assert rec["aligned_dims"] == 0                       # pure beta-VAE (D-022)
    assert "alignment" not in rec                         # no head -> no R2 block
    assert np.isfinite(rec["interpretability"]["mig"])    # still scored vs constructs
    assert rec["loss_final"]["align"] == 0.0


def test_run_tag_is_stable():
    assert _run_tag(3, 0.05, 4.0) == "s3_b0.05_l4"
    assert _run_tag(0, 1.0, 0.0) == "s0_b1_l0"
