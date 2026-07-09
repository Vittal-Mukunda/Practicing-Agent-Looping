"""Smoke test for the Phase 3 evaluation pipeline: the full frozen-representation
protocol (AE + all baselines + both heads + metrics) runs end-to-end on tiny
synthetic data (CPU, 2 epochs) and emits sane metric records for every baseline.
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from cadvae.data.constructs import ConstructSet
from cadvae.data.personality import PreparedSplit
from cadvae.eval.protocol import evaluate_baselines
from cadvae.eval.stats import aggregate, significance_vs_best

METHODS = {"rfm", "rfm_kmeans", "ae", "ae_kmeans", "gmm_ae", "dec"}


def _tiny(seed=0):
    rng = np.random.default_rng(seed)
    ntr, nte, d, c = 200, 80, 6, 5
    ytr = (rng.random(ntr) < 0.2).astype(np.int64)
    yte = (rng.random(nte) < 0.2).astype(np.int64)
    ps = PreparedSplit(
        X_train=rng.standard_normal((ntr, d)), X_val=rng.standard_normal((10, d)),
        X_test=rng.standard_normal((nte, d)),
        y_train=ytr, y_val=(rng.random(10) < 0.2).astype(np.int64), y_test=yte,
        feature_names=[f"f{i}" for i in range(d)], ids={},
    )
    ks = ConstructSet(
        C_train=rng.standard_normal((ntr, c)), C_val=rng.standard_normal((10, c)),
        C_test=rng.standard_normal((nte, c)),
        names=["rfm_R", "rfm_F", "rfm_M", "price_sensitivity", "affinity_x"], raw={},
    )
    return ps, ks


def _cfg():
    return OmegaConf.create({
        "device": "cpu",
        "model": {"latent_dim": 4, "hidden_dims": [8], "lr": 1e-3,
                  "batch_size": 64, "max_epochs": 2},
        "eval": {"n_clusters": 3, "train_subsample": None, "k_frac": 0.10,
                 "primary_metric": "pr_auc"},
    })


def test_evaluate_baselines_runs_and_is_sane():
    ps, ks = _tiny()
    recs = evaluate_baselines(ps, ks, _cfg(), seed=0, device="cpu")
    assert {r["method"] for r in recs} == METHODS
    for r in recs:
        assert r["head"] in ("logreg", "gbt")
        for m in ("roc_auc", "pr_auc"):
            assert 0.0 <= r[m] <= 1.0


def test_aggregate_and_significance_over_seeds():
    ps0, ks0 = _tiny(0)
    ps1, ks1 = _tiny(1)
    recs = evaluate_baselines(ps0, ks0, _cfg(), 0, "cpu") + \
        evaluate_baselines(ps1, ks1, _cfg(), 1, "cpu")
    summary = aggregate(recs)
    assert all(r["n_seeds"] == 2 for r in summary)
    w = significance_vs_best(recs, metric="pr_auc", head="gbt")
    assert w["best_method"] in METHODS
    assert 0.0 <= w["best_mean"] <= 1.0
    # both significance families are reported vs the best baseline (CLAUDE.md §statistics)
    assert w["n_seeds"] == 2
    assert w["best_method"] not in w["ttest_p"]  # the best is not compared against itself
    others = METHODS - {w["best_method"]}
    assert set(w["wilcoxon_p"]) == set(w["ttest_p"]) == others
    # n=2 Wilcoxon floor is 2^-(n-1) = 0.5 — it cannot approach 0.05; the t-test carries it
    assert w["wilcoxon_floor"] == 0.5
