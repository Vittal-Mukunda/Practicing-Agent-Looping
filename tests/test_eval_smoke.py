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

METHODS = {"raw", "pca", "rfm", "rfm_kmeans", "ae", "ae_kmeans", "gmm_ae", "dec"}


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


def _tiny_tasks(ps, seed=0):
    """Extra binary + multiclass tasks aligned to the tiny split (D-023)."""
    rng = np.random.default_rng(seed + 100)
    ntr, nte = len(ps.y_train), len(ps.y_test)
    return {
        "churned": {"type": "binary",
                    "train": ((rng.random(ntr) < 0.5).astype(np.int64), np.ones(ntr, bool)),
                    "test": ((rng.random(nte) < 0.5).astype(np.int64), np.ones(nte, bool))},
        "next_category": {"type": "multiclass", "top_ks": (1, 3),
                          "train": (rng.integers(0, 4, ntr), rng.random(ntr) < 0.6),
                          "test": (rng.integers(0, 4, nte), rng.random(nte) < 0.6)},
    }


def _cfg(methods=None):
    return OmegaConf.create({
        "device": "cpu",
        "model": {"latent_dim": 4, "hidden_dims": [8], "lr": 1e-3,
                  "batch_size": 64, "max_epochs": 2},
        "eval": {"n_clusters": 3, "train_subsample": None, "k_frac": 0.10,
                 "primary_metric": "pr_auc", "methods": methods},
    })


def test_evaluate_baselines_runs_and_is_sane():
    ps, ks = _tiny()
    recs = evaluate_baselines(ps, ks, _cfg(), seed=0, device="cpu")
    assert {r["method"] for r in recs} == METHODS
    for r in recs:
        assert r["head"] in ("logreg", "gbt")
        assert r["task"] == "primary"
        for m in ("roc_auc", "pr_auc"):
            assert 0.0 <= r[m] <= 1.0


def test_methods_filter_and_unknown_method_rejected():
    import pytest
    ps, ks = _tiny()
    recs = evaluate_baselines(ps, ks, _cfg(methods=["raw", "rfm"]), seed=0, device="cpu")
    assert {r["method"] for r in recs} == {"raw", "rfm"}
    with pytest.raises(ValueError, match="unknown eval.methods"):
        evaluate_baselines(ps, ks, _cfg(methods=["rmf"]), seed=0, device="cpu")


def test_multitask_records_binary_and_multiclass():
    ps, ks = _tiny()
    recs = evaluate_baselines(ps, ks, _cfg(methods=["raw", "rfm"]), seed=0,
                              device="cpu", tasks=_tiny_tasks(ps))
    tasks = {r["task"] for r in recs}
    assert tasks == {"primary", "churned", "next_category"}
    mc = [r for r in recs if r["task"] == "next_category"]
    for r in mc:
        assert 0.0 <= r["acc_at_1"] <= r["acc_at_3"] <= 1.0   # top-k is monotone
        assert "pr_auc" not in r                              # binary metrics absent
    ch = [r for r in recs if r["task"] == "churned"]
    assert all(0.0 <= r["pr_auc"] <= 1.0 for r in ch)


def test_multiclass_head_survives_singleton_class_above_early_stop_threshold():
    """Regression: >10k samples used to flip HistGBT early_stopping='auto' ON, whose
    stratified internal split raises on a 1-member class (hit on unsubsampled B)."""
    from cadvae.eval.protocol import _fit_eval_heads_multiclass
    rng = np.random.default_rng(0)
    n = 10_050                                   # just over the auto threshold
    R = rng.standard_normal((n, 3))
    y = rng.integers(0, 3, n)
    y[0] = 3                                     # singleton class
    m = _fit_eval_heads_multiclass(R[:n // 2], y[:n // 2], R[n // 2:], y[n // 2:],
                                   seed=0, top_ks=(1,))
    assert 0.0 <= m["gbt"]["acc_at_1"] <= 1.0


def test_aggregate_and_significance_over_seeds():
    ps0, ks0 = _tiny(0)
    ps1, ks1 = _tiny(1)
    recs = evaluate_baselines(ps0, ks0, _cfg(), 0, "cpu", tasks=_tiny_tasks(ps0, 0)) + \
        evaluate_baselines(ps1, ks1, _cfg(), 1, "cpu", tasks=_tiny_tasks(ps1, 1))
    summary = aggregate(recs)
    assert all(r["n_seeds"] == 2 for r in summary)
    assert {r["task"] for r in summary} == {"primary", "churned", "next_category"}
    w = significance_vs_best(recs, metric="pr_auc", head="gbt")
    assert w["best_method"] in METHODS
    # raw is the reference ceiling: compared against, but never allowed to BE the bar
    assert w["best_method"] != "raw"
    assert "raw" in w["ttest_p"]
    assert 0.0 <= w["best_mean"] <= 1.0
    # both significance families are reported vs the best baseline (CLAUDE.md §statistics)
    assert w["n_seeds"] == 2
    assert w["best_method"] not in w["ttest_p"]  # the best is not compared against itself
    others = METHODS - {w["best_method"]}
    assert set(w["wilcoxon_p"]) == set(w["ttest_p"]) == others
    # n=2 Wilcoxon floor is 2^-(n-1) = 0.5 — it cannot approach 0.05; the t-test carries it
    assert w["wilcoxon_floor"] == 0.5
