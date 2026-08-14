"""Ground-truth tests for the multiplicity correction and interval estimates (D-038).

The project reports many paired comparisons whose Wilcoxon p-values sit exactly at
the n=6 discrete floor of 0.03125. Whether those survive as a family is the single
statistical question a reviewer will check, so the correction has to be right:
step-down monotone, correct family size, NaN-safe, and it must actually reject the
floor value once the family is large enough.
"""

from __future__ import annotations

import numpy as np

from cadvae.eval.stats import bootstrap_ci, holm_bonferroni, paired_effect_size

WILCOXON_FLOOR_6 = 2.0 ** -5      # 0.03125


def test_holm_matches_the_textbook_procedure():
    p = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.60}
    out = holm_bonferroni(p, alpha=0.05)
    assert out["m"] == 4
    # (m-i)*p, then enforced monotone non-decreasing
    assert np.isclose(out["adjusted"]["a"], 0.04)      # 4 * 0.01
    assert np.isclose(out["adjusted"]["b"], 0.06)      # 3 * 0.02
    assert np.isclose(out["adjusted"]["c"], 0.06)      # 2 * 0.03 = 0.06, ties the step
    assert np.isclose(out["adjusted"]["d"], 0.60)      # 1 * 0.60
    assert out["reject"] == {"a": True, "b": False, "c": False, "d": False}
    assert out["n_reject"] == 1


def test_adjusted_p_values_are_monotone_in_the_sorted_order():
    rng = np.random.default_rng(0)
    p = {f"m{i}": float(v) for i, v in enumerate(rng.uniform(0, 1, 25))}
    adj = holm_bonferroni(p)["adjusted"]
    ordered = [adj[k] for k in sorted(p, key=lambda k: p[k])]
    assert all(b >= a - 1e-12 for a, b in zip(ordered, ordered[1:], strict=False))
    assert max(ordered) <= 1.0


def test_the_wilcoxon_floor_stops_being_significant_as_the_family_grows():
    """The reason this correction is not optional for this project."""
    assert holm_bonferroni({"x": WILCOXON_FLOOR_6})["reject"]["x"] is True
    # a family of two floor-valued comparisons already fails at alpha = 0.05
    two = holm_bonferroni({f"m{i}": WILCOXON_FLOOR_6 for i in range(2)})
    assert two["n_reject"] == 0
    assert np.isclose(two["adjusted"]["m0"], 0.0625)
    # the realistic family (8 challengers) is nowhere near
    assert holm_bonferroni({f"m{i}": WILCOXON_FLOOR_6 for i in range(8)})["n_reject"] == 0


def test_nan_p_values_are_excluded_from_the_family_size():
    """Degenerate comparisons (identical vectors) must not inflate m and weaken others."""
    out = holm_bonferroni({"a": 0.01, "b": float("nan"), "c": float("nan")})
    assert out["m"] == 1
    assert np.isclose(out["adjusted"]["a"], 0.01)
    assert out["reject"] == {"a": True, "b": False, "c": False}
    assert np.isnan(out["adjusted"]["b"])
    empty = holm_bonferroni({"a": float("nan")})
    assert empty["m"] == 0 and empty["n_reject"] == 0


def test_effect_size_signs_and_magnitude():
    a = np.array([0.60, 0.62, 0.61, 0.63, 0.59, 0.60])
    b = a - 0.05
    e = paired_effect_size(a, b)
    assert np.isclose(e["mean_diff"], 0.05) and e["n"] == 6
    assert np.isnan(e["cohens_dz"])                      # constant difference, sd = 0
    e2 = paired_effect_size(a, b + np.array([0, .01, -.01, .02, -.02, 0]))
    assert e2["mean_diff"] > 0 and e2["cohens_dz"] > 1.0
    assert paired_effect_size(b, a)["mean_diff"] < 0      # antisymmetric


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def test_bootstrap_ci_brackets_the_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 800)
    s = 0.35 * y + rng.normal(size=800)                  # informative but noisy
    ci = bootstrap_ci(y, s, _auc, n_boot=400, seed=7)
    assert ci["lo"] < ci["point"] < ci["hi"]
    assert ci["n_boot"] + ci["skipped"] == 400
    assert bootstrap_ci(y, s, _auc, n_boot=400, seed=7) == ci        # deterministic
    assert bootstrap_ci(y, s, _auc, n_boot=400, seed=8) != ci        # seed actually used


def test_bootstrap_ci_is_wider_on_a_small_test_set():
    """Dataset A's 448-row test split is exactly why the interval is reported."""
    rng = np.random.default_rng(1)
    big_y = rng.integers(0, 2, 4000)
    big_s = 0.35 * big_y + rng.normal(size=4000)
    wide = bootstrap_ci(big_y[:300], big_s[:300], _auc, n_boot=400, seed=0)
    tight = bootstrap_ci(big_y, big_s, _auc, n_boot=400, seed=0)
    assert (wide["hi"] - wide["lo"]) > 2 * (tight["hi"] - tight["lo"])


def test_single_class_resamples_are_skipped_not_counted():
    y = np.array([1] + [0] * 40)                          # 1 positive → many degenerate draws
    s = np.linspace(0, 1, 41)
    ci = bootstrap_ci(y, s, _auc, n_boot=200, seed=0)
    assert ci["skipped"] > 0
    assert ci["n_boot"] == 200 - ci["skipped"]
