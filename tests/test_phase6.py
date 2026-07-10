"""Tests for the Phase-6 analysis stack (sweep_analysis, stability, figures) and
the Phase-7 reproduction comparator — synthetic data only, no dataset I/O.

Every function that the Phase-6 orchestrator calls is exercised here on inputs
shaped exactly like the Phase-5 sweep output, so the orchestrator's parts are
verified BEFORE any sweep exists (D-030/031/032 pre-implementation, owner request
2026-07-10)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cadvae.eval.repro_check import compare_records
from cadvae.eval.stability import (
    bootstrap_persistence,
    kmeans_assign,
    load_model_state,
    pairwise_stability,
    rebuild_cadvae,
    stability_sample,
)
from cadvae.eval.sweep_analysis import (
    aggregate_sweep,
    flatten_record,
    paired_tests,
    pareto_frontier,
    per_seed_values,
    select_sweet_spot,
)
from cadvae.models.cadvae import CADVAE
from cadvae.viz.figures import markdown_table, persona_cards, stability_bars, tradeoff_curve


# ---------------------------------------------------------------- sweep records
def _fake_record(seed, beta, lam, pr, mig, with_next=True):
    """Mimics exactly the JSON written by run_cadvae_sweep.run_point."""
    heads = {"logreg": {"roc_auc": 0.7, "pr_auc": pr - 0.01, "prec_at_10pct": 0.3,
                        "base_rate": 0.15},
             "gbt": {"roc_auc": 0.72, "pr_auc": pr, "prec_at_10pct": 0.32,
                     "base_rate": 0.15}}
    downstream = {"primary": heads,
                  "churned": {"logreg": {"pr_auc": 0.8, "roc_auc": 0.7,
                                         "prec_at_10pct": 0.9, "base_rate": 0.7},
                              "gbt": {"pr_auc": 0.82, "roc_auc": 0.71,
                                      "prec_at_10pct": 0.91, "base_rate": 0.7}}}
    if with_next:
        mc = {"acc_at_1": 0.5, "acc_at_3": 0.8, "acc_at_5": 0.9,
              "macro_ovr_auc": 0.6, "base_rate": 0.5, "n_classes": 12}
        downstream["next_category"] = {"logreg": dict(mc), "gbt": dict(mc)}
    val_heads = {"logreg": {"roc_auc": 0.69, "pr_auc": pr - 0.012, "prec_at_10pct": 0.3,
                            "base_rate": 0.15},
                 "gbt": {"roc_auc": 0.71, "pr_auc": pr - 0.003, "prec_at_10pct": 0.31,
                         "base_rate": 0.15}}
    rec = {"dataset": "fake", "seed": seed, "beta": beta, "lambda_align": lam,
           "aligned_dims": 3 if lam > 0 else 0,
           "loss_final": {"total": 30.0, "recon": 25.0, "kl_free": 0.5,
                          "kl_aligned": 2.0, "align": 3.0},
           "loss_history": [],
           "downstream": downstream,
           "downstream_val": {"primary": val_heads},
           "interpretability": {"mig": mig, "sap": mig / 2,
                                "mig_per_factor": {"c0": mig},
                                "sap_per_factor": {"c0": mig / 2},
                                "axis_alignment": {"c0": {"best_dim": 0, "best_r2": 0.9}}},
           "wall_time_s": 1.0}
    if lam > 0:
        rec["alignment"] = {"r2_per_construct": {"c0": 0.9}, "r2_mean": 0.85,
                            "r2_rfm_mean": 0.9}
    return rec


def _fake_sweep():
    """2 betas x 2 lambdas x 3 seeds; higher lambda -> higher mig, lower pr."""
    recs = []
    for seed in (0, 1, 2):
        for beta in (0.1, 1.0):
            for lam in (0.0, 1.0):
                pr = 0.50 - 0.05 * lam - 0.02 * beta + 0.001 * seed
                mig = 0.05 + 0.30 * lam + 0.01 * beta
                recs.append(_fake_record(seed, beta, lam, pr, mig))
    return recs


def test_flatten_and_aggregate_nan_aware():
    recs = _fake_sweep()
    flat = flatten_record(recs[0])
    assert {"pr_auc", "churned_pr_auc", "next_category_acc_at_1", "mig",
            "r2_mean", "kl_free", "val_pr_auc"} <= set(flat)
    assert flat["val_pr_auc"] != flat["pr_auc"]     # val metrics are their own numbers
    agg = aggregate_sweep(recs)
    assert len(agg) == 4 and all(a["n_seeds"] == 3 for a in agg)
    lam0 = [a for a in agg if a["lambda_align"] == 0.0]
    assert all(np.isnan(a["r2_mean_mean"]) for a in lam0)   # beta-VAE: no align head
    assert all(np.isfinite(a["pr_auc_mean"]) for a in agg)


def test_pareto_and_sweet_spot():
    agg = aggregate_sweep(_fake_sweep())
    front = pareto_frontier(agg, "mig_mean", "pr_auc_mean")
    assert 1 <= len(front) <= len(agg)
    sel = select_sweet_spot(agg, y_slack=1.0)     # huge slack -> max-mig frontier point
    assert sel["point"]["lambda_align"] == 1.0
    sel_tight = select_sweet_spot(agg, y_slack=0.0)
    best_pr = max(a["pr_auc_mean"] for a in agg)
    assert sel_tight["point"]["pr_auc_mean"] >= best_pr - 1e-12


def test_sweet_spot_raises_without_finite_y():
    with pytest.raises(ValueError, match="finite"):
        select_sweet_spot([{"beta": 1.0, "lambda_align": 0.0,
                            "pr_auc_mean": float("nan"), "mig_mean": 0.1}])


def test_per_seed_values_and_paired_tests():
    recs = _fake_sweep()
    a = per_seed_values(recs, 0.1, 1.0, metric="pr_auc")
    assert set(a) == {0, 1, 2}
    b = {s: v + 0.05 + 0.002 * s for s, v in a.items()}   # non-constant shift
    t = paired_tests(a, b)
    assert t["n_seeds"] == 3 and t["ttest_p"] < 0.05
    same = paired_tests(a, dict(a))               # zero differences: p=1, no crash
    assert same["wilcoxon_p"] == 1.0 and same["ttest_p"] == 1.0
    tiny = paired_tests({0: 1.0}, {0: 2.0})       # n=1 -> NaN p-values
    assert np.isnan(tiny["ttest_p"])


# ------------------------------------------------------------------- stability
def test_pairwise_stability_permutation_invariant_and_bounds():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 500)
    perm = np.array([2, 3, 0, 1])[labels]          # same partition, renamed clusters
    s = pairwise_stability({0: labels, 1: perm, 2: labels})
    assert s["ari_mean"] == pytest.approx(1.0)
    assert s["nmi_mean"] == pytest.approx(1.0)
    noise = rng.integers(0, 4, 500)
    s2 = pairwise_stability({0: labels, 1: noise})
    assert s2["ari_mean"] < 0.1                    # independent partitions ~ chance


def test_pairwise_stability_guards():
    one = {0: np.zeros(10, dtype=np.int64)}
    s = pairwise_stability(one)
    assert s["n_pairs"] == 0 and np.isnan(s["ari_mean"])
    with pytest.raises(ValueError, match="lengths differ"):
        pairwise_stability({0: np.zeros(10, dtype=np.int64),
                            1: np.zeros(9, dtype=np.int64)})


def test_bootstrap_persistence_high_on_separated_blobs():
    rng = np.random.default_rng(1)
    Z = np.vstack([rng.normal(c * 8.0, 0.4, (80, 3)) for c in range(3)])
    b = bootstrap_persistence(Z, k=3, n_boot=5, seed=0)
    assert b["ari_mean"] > 0.95                    # well-separated -> persistent
    with pytest.raises(ValueError, match="cannot fit"):
        bootstrap_persistence(Z[:2], k=3, n_boot=2, seed=0)


def test_kmeans_assign_deterministic_and_guarded():
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((100, 4))
    assert np.array_equal(kmeans_assign(Z, 3, seed=5), kmeans_assign(Z, 3, seed=5))
    with pytest.raises(ValueError, match="cannot fit"):
        kmeans_assign(Z[:2], 3, seed=0)


def test_stability_sample_fixed_and_capped():
    a = stability_sample(1000, 100, sample_seed=7)
    assert np.array_equal(a, stability_sample(1000, 100, sample_seed=7))
    assert len(a) == 100 and len(np.unique(a)) == 100
    assert np.array_equal(stability_sample(50, 100, sample_seed=7), np.arange(50))


# -------------------------------------------------- model rebuild + persona card
def _tiny_state(tmp_path, aligned=2, latent=5, in_dim=7):
    model = CADVAE(in_dim, latent, [12], aligned, 3 if aligned else 0)
    state = {"state_dict": model.state_dict(), "in_dim": in_dim, "latent_dim": latent,
             "hidden_dims": [12], "aligned_dims": aligned,
             "n_constructs": 3 if aligned else 0,
             "seed": 0, "beta": 1.0, "lambda_align": 1.0 if aligned else 0.0}
    p = tmp_path / "m.pt"
    torch.save(state, p)
    return p


def test_model_state_roundtrip_and_rebuild(tmp_path):
    p = _tiny_state(tmp_path)
    state = load_model_state(p)                    # weights_only=True path must work
    model = rebuild_cadvae(state)
    mu, logvar = model.encode(torch.zeros(4, 7))
    assert mu.shape == (4, 5) and not model.training


class _FakeScaler:
    def inverse_transform(self, X):
        return X * 2.0 + 1.0


def test_persona_cards_render(tmp_path):
    state = load_model_state(_tiny_state(tmp_path))
    out = persona_cards(state, _FakeScaler(), [f"feat{i}" for i in range(7)],
                        tmp_path / "cards.png", axis_labels={0: "rfm_R (R2=0.90)"},
                        top_n=5)
    assert out.exists() and out.stat().st_size > 0


def test_persona_cards_reject_pure_beta_vae(tmp_path):
    state = load_model_state(_tiny_state(tmp_path, aligned=0))
    with pytest.raises(ValueError, match="no aligned dims"):
        persona_cards(state, _FakeScaler(), [f"f{i}" for i in range(7)],
                      tmp_path / "x.png")


# --------------------------------------------------------------------- figures
def test_tradeoff_curve_renders_with_bar_ceiling_sweet(tmp_path):
    agg = aggregate_sweep(_fake_sweep())
    sweet = select_sweet_spot(agg, y_slack=1.0)["point"]
    out = tradeoff_curve(agg, tmp_path / "t.png", bar={"name": "rfm", "value": 0.45},
                         ceiling={"name": "raw", "value": 0.52}, sweet=sweet)
    assert out.exists() and out.stat().st_size > 0
    with pytest.raises(ValueError, match="nothing to plot"):
        tradeoff_curve([{"beta": 1, "lambda_align": 0, "mig_mean": float("nan"),
                         "pr_auc_mean": float("nan")}], tmp_path / "e.png")


def test_stability_bars_and_table(tmp_path):
    res = {"cadvae_sweet": {"ari_mean": 0.8, "ari_std": 0.05, "nmi_mean": 0.7,
                            "nmi_std": 0.04},
           "beta_vae": {"ari_mean": 0.5, "ari_std": 0.2, "nmi_mean": 0.45,
                        "nmi_std": 0.15}}
    out = stability_bars(res, tmp_path / "s.png")
    assert out.exists() and out.stat().st_size > 0
    t = markdown_table([{"config": "full", "pr_auc": 0.5123, "mig": 0.31}],
                       tmp_path / "t.md", title="Ablations")
    text = t.read_text()
    assert "| config | pr_auc | mig |" in text and "0.5123" in text


def test_best_baseline_for_task_picks_per_task_winner(tmp_path):
    import json

    from cadvae.eval.sweep_analysis import best_baseline_for_task
    recs = []
    for s in (0, 1):
        recs += [{"method": "rfm", "head": "gbt", "seed": s, "task": "primary",
                  "pr_auc": 0.45},
                 {"method": "ae", "head": "gbt", "seed": s, "task": "primary",
                  "pr_auc": 0.40},
                 {"method": "rfm", "head": "gbt", "seed": s, "task": "churned",
                  "pr_auc": 0.80},
                 {"method": "ae", "head": "gbt", "seed": s, "task": "churned",
                  "pr_auc": 0.86},
                 {"method": "raw", "head": "gbt", "seed": s, "task": "churned",
                  "pr_auc": 0.99}]
    p = tmp_path / "records.json"
    p.write_text(json.dumps(recs))
    assert best_baseline_for_task(p, "primary", "pr_auc")["method"] == "rfm"
    churn = best_baseline_for_task(p, "churned", "pr_auc")
    assert churn["method"] == "ae"                 # per-task winner, raw excluded
    assert set(churn["by_seed"]) == {0, 1}
    assert best_baseline_for_task(p, "nope", "pr_auc") is None


def test_load_sweep_records_names_corrupt_file(tmp_path):
    import json

    from cadvae.eval.sweep_analysis import load_sweep_records
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "s0_b1_l1.json").write_text(json.dumps({"ok": 1}))
    (runs / "s1_b1_l1.json").write_text('{"beta": 1.0, "lam')   # truncated mid-write
    with pytest.raises(ValueError, match="s1_b1_l1"):
        load_sweep_records(tmp_path)


# ----------------------------------------------- orchestrator end-to-end (faked)
def test_run_phase6_end_to_end_on_faked_artifacts(tmp_path, monkeypatch):
    """Full analyze() against a synthetic sweep dir + phase-3 dir: the exact glue
    that will run after the real sweep, minus any compute."""
    import json

    from omegaconf import OmegaConf

    import cadvae.eval.run_phase6 as p6
    from cadvae.data.constructs import ConstructSet
    from cadvae.data.personality import PreparedSplit
    from cadvae.eval.run_cadvae_sweep import _run_tag

    in_dim, latent, aligned = 7, 5, 2
    betas, lams, seeds = (0.1, 1.0), (0.0, 1.0), (0, 1, 2)

    sweep = tmp_path / "results" / "phase5" / "fake_sweep"
    (sweep / "runs").mkdir(parents=True)
    (sweep / "models").mkdir()
    for s in seeds:
        for b in betas:
            for lam in lams:
                pr = 0.50 - 0.05 * lam - 0.02 * b + 0.001 * s
                rec = _fake_record(s, b, lam, pr, 0.05 + 0.3 * lam)
                (sweep / "runs" / f"{_run_tag(s, b, lam)}.json").write_text(json.dumps(rec))
                model = CADVAE(in_dim, latent, [12], aligned if lam > 0 else 0,
                               3 if lam > 0 else 0)
                torch.save({"state_dict": model.state_dict(), "in_dim": in_dim,
                            "latent_dim": latent, "hidden_dims": [12],
                            "aligned_dims": aligned if lam > 0 else 0,
                            "n_constructs": 3 if lam > 0 else 0, "seed": s,
                            "beta": b, "lambda_align": lam},
                           sweep / "models" / f"{_run_tag(s, b, lam)}.pt")

    p3 = tmp_path / "results" / "phase3" / "fake_baselines"
    p3.mkdir(parents=True)
    recs3 = []
    for s in seeds:
        recs3 += [
            {"method": "rfm", "head": "gbt", "seed": s, "task": "primary",
             "roc_auc": 0.7, "pr_auc": 0.45 + 0.001 * s},
            {"method": "raw", "head": "gbt", "seed": s, "task": "primary",
             "roc_auc": 0.75, "pr_auc": 0.52},
            {"method": "rfm", "head": "gbt", "seed": s, "task": "churned",
             "roc_auc": 0.6, "pr_auc": 0.80},
            {"method": "ae", "head": "gbt", "seed": s, "task": "churned",
             "roc_auc": 0.65, "pr_auc": 0.86 + 0.001 * s},   # per-task winner != rfm
            {"method": "rfm", "head": "gbt", "seed": s, "task": "next_category",
             "acc_at_1": 0.40, "macro_ovr_auc": 0.52},
        ]
    (p3 / "records.json").write_text(json.dumps(recs3))
    (p3 / "summary.json").write_text(json.dumps(
        [{"task": "primary", "method": "raw", "head": "gbt", "n_seeds": 3,
          "pr_auc_mean": 0.52, "pr_auc_std": 0.001}]))
    (p3 / "significance.json").write_text(json.dumps(
        {"best_method": "rfm", "best_mean": 0.4510}))

    rng = np.random.default_rng(0)
    n = 90
    X = rng.standard_normal((n, in_dim))
    y = (rng.random(n) < 0.2).astype(np.int64)
    C = rng.standard_normal((n, 3))

    class _Scaler:
        def inverse_transform(self, A):
            return A * 2.0 + 1.0

    def fake_prepare(cfg, seed):
        third = n // 3
        ps = PreparedSplit(
            X_train=X[:third], X_val=X[third:2 * third], X_test=X[2 * third:],
            y_train=y[:third], y_val=y[third:2 * third], y_test=y[2 * third:],
            feature_names=[f"f{i}" for i in range(in_dim)],
            ids={"train": np.arange(third), "val": np.arange(third, 2 * third),
                 "test": np.arange(2 * third, n)},
            fitted={"scaler": _Scaler()})
        ks = ConstructSet(C_train=C[:third], C_val=C[third:2 * third],
                          C_test=C[2 * third:], names=["c0", "c1", "c2"],
                          raw={})
        return ps, ks

    monkeypatch.setattr(p6, "_prepare", fake_prepare)
    cfg = OmegaConf.create({
        "seed": 7, "device": "cpu", "results_dir": str(tmp_path / "results"),
        "data": {"name": "fake"},
        "eval": {"seeds": list(seeds), "n_clusters": 3, "train_subsample": None,
                 "k_frac": 0.10, "primary_metric": "pr_auc", "methods": None},
        "phase6": {"head": "gbt", "x_metric": "mig_mean", "select_on": "val_pr_auc",
                   "y_slack": None, "stability_users": 50,
                   "stability_seed": 20260710, "n_boot": 3,
                   "traversal_span": 2.0, "card_top_features": 5},
    })
    analysis = p6.analyze(cfg)

    out = tmp_path / "results" / "phase6" / "fake_analysis"
    assert (out / "analysis.json").exists()
    json.dumps(analysis)                                    # fully serializable
    assert analysis["bar"] == {"name": "rfm", "value": 0.4510}
    # sweet spot selected on VALIDATION (D-033), never on the test metric
    assert analysis["sweet_spot"]["rule"]["y"] == "val_pr_auc_mean"
    sig = analysis["significance_vs_bar"]
    assert {"primary", "churned/pr_auc", "next_category/acc_at_1",
            "next_category/macro_ovr_auc"} <= set(sig)
    assert sig["churned/pr_auc"]["baseline"] == "ae"        # best PER-TASK baseline
    assert sig["primary"]["baseline"] == "rfm"
    assert analysis["stability"], "stability must cover at least one config"
    for v in analysis["stability"].values():
        assert -1.0 <= v["ari_mean"] <= 1.0 and v["n_users"] == 50
        assert np.isfinite(v["clustering"]["silhouette"])   # CLAUDE.md protocol
    assert (out / "figures" / "tradeoff_mig.png").exists()
    assert (out / "tables" / "ablations.md").exists()
    assert (out / "tables" / "multitask.md").exists()
    # persona cards render unless the sweet spot landed on lambda=0
    sl = analysis["sweet_spot"]["point"]["lambda_align"]
    if sl > 0:
        assert (out / "figures" / "persona_cards.png").exists()


# ------------------------------------------------------------------ repro check
def test_compare_records_pass_tolerance_and_mismatch():
    ref = [{"method": "rfm", "head": "gbt", "seed": 0, "task": "primary",
            "roc_auc": 0.75, "pr_auc": 0.20}]
    same = [dict(ref[0])]
    assert compare_records(same, ref) == []
    close = [{**ref[0], "pr_auc": 0.20 + 1e-12}]
    assert compare_records(close, ref, atol=1e-9) == []
    off = [{**ref[0], "pr_auc": 0.21}]
    problems = compare_records(off, ref, atol=1e-9)
    assert len(problems) == 1 and "pr_auc" in problems[0]
    missing = [{**ref[0], "method": "ae"}]
    assert "not present" in compare_records(missing, ref)[0]
    nans = [{**ref[0], "pr_auc": float("nan")}]
    ref_nan = [{**ref[0], "pr_auc": float("nan")}]
    assert compare_records(nans, ref_nan) == []    # NaN == NaN for reproduction
