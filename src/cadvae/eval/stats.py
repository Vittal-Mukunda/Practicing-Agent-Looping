"""Aggregate multi-seed records → mean ± std + paired significance.

CLAUDE.md statistical-rigor rule: every headline number is mean ± std over ≥5 seeds,
plus a paired significance test vs the strongest baseline (never a single run).
Records are grouped by (task, method, head); metrics are auto-detected per group so
binary (roc/pr/prec@k) and multiclass (acc@k/macro-AUC) tasks aggregate uniformly.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# record keys that are identifiers, not metrics
_META_KEYS = {"method", "head", "seed", "task"}


def _metric_keys(rs: list[dict]) -> list[str]:
    """Numeric metric keys shared by every record in a group (order-stable)."""
    keys = [k for k in rs[0] if k not in _META_KEYS
            and isinstance(rs[0][k], (int, float, np.floating))]
    return [k for k in keys if all(k in r for r in rs)]


def aggregate(records: list[dict]) -> list[dict]:
    """Mean ± std of each metric per (task, method, head) across seeds."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r.get("task", "primary"), r["method"], r["head"])].append(r)
    out = []
    for (task, method, head), rs in groups.items():
        row = {"task": task, "method": method, "head": head, "n_seeds": len(rs)}
        for m in _metric_keys(rs):
            vals = np.array([r[m] for r in rs], dtype=float)
            row[f"{m}_mean"] = float(np.nanmean(vals))
            row[f"{m}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def _by_seed(records, method, head, metric, task):
    return {r["seed"]: r[metric] for r in records
            if r["method"] == method and r["head"] == head
            and r.get("task", "primary") == task}


def significance_vs_best(records: list[dict], metric: str = "pr_auc",
                         head: str = "gbt", task: str = "primary",
                         exclude_from_best: tuple[str, ...] = ("raw",)) -> dict:
    """Paired significance of every method's per-seed metric vs the best-mean method.

    ``exclude_from_best``: methods that may never define the bar — ``raw`` (the full
    feature vector) is a reference ceiling, not a persona representation (D-024); it
    is still *compared against* the bar like everything else.

    Reports BOTH a paired Wilcoxon signed-rank and a paired t-test (CLAUDE.md allows
    either). NOTE: two-sided Wilcoxon has a discrete p-floor of 2^-(n-1) — at n=5
    seeds that floor is 0.0625, so it can never reach p<0.05 no matter how consistent
    the win. At the D-023 six seeds the floor is 0.03125, so both tests can carry a
    significance claim; ``n_seeds`` is reported so the reader can judge.
    """
    sel = [r for r in records if r["head"] == head and r.get("task", "primary") == task]
    methods = sorted({r["method"] for r in sel})
    means = {m: np.mean([r[metric] for r in sel if r["method"] == m]) for m in methods}
    eligible = [m for m in methods if m not in exclude_from_best] or methods
    best = max(eligible, key=lambda m: means[m])
    best_by_seed = _by_seed(records, best, head, metric, task)
    wilcox, tt, effects = {}, {}, {}
    n_used = 0
    for m in methods:
        if m == best:
            continue
        mb = _by_seed(records, m, head, metric, task)
        seeds = sorted(set(best_by_seed) & set(mb))
        n_used = len(seeds)
        a = np.array([best_by_seed[s] for s in seeds])
        b = np.array([mb[s] for s in seeds])
        effects[m] = paired_effect_size(a, b)
        if len(seeds) < 2 or np.allclose(a, b):
            wilcox[m] = tt[m] = float("nan")
        else:
            wilcox[m] = float(wilcoxon(a, b).pvalue)
            tt[m] = float(ttest_rel(a, b).pvalue)
    return {"best_method": best, "best_mean": float(means[best]), "metric": metric,
            "head": head, "task": task, "n_seeds": n_used,
            "excluded_from_best": list(exclude_from_best),
            "wilcoxon_floor": (2.0 ** -(n_used - 1) if n_used >= 2 else float("nan")),
            "wilcoxon_p": wilcox, "ttest_p": tt,
            # familywise correction over the comparisons made HERE (one bar, one
            # head, one task, all challengers) — D-038. A caller comparing across
            # several tasks/heads must re-run holm_bonferroni over the union.
            "holm_wilcoxon": holm_bonferroni(wilcox),
            "holm_ttest": holm_bonferroni(tt),
            "effect_size": effects}


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm step-down familywise correction over a DECLARED family of comparisons.

    Why this is not optional here (D-038). The comparison family is large — methods
    x heads x tasks, and again per sweep grid point — while the two-sided Wilcoxon
    signed-rank p-value at n = 6 seeds has a discrete floor of 2^-(n-1) = 0.03125.
    Every Wilcoxon "win" in this project therefore sits AT the floor, and an
    uncorrected floor value cannot survive a family of more than ``alpha/floor`` ~ 1
    comparison. Reporting raw floor p-values across a family of 16+ comparisons and
    calling them significant is the exact error a statistically-minded reviewer
    looks for. Holm is used rather than Bonferroni because it is uniformly more
    powerful at the same familywise error rate and needs no independence assumption.

    Holm, "A simple sequentially rejective multiple test procedure", Scandinavian
    Journal of Statistics 6(2):65-70, 1979.

    NaN p-values (degenerate comparisons: identical vectors, <2 paired seeds) are
    excluded from the family size ``m`` and returned as NaN.

    Returns ``{"m", "alpha", "adjusted": {name: p_adj}, "reject": {name: bool},
    "n_reject"}``, where ``p_adj`` is the standard monotone step-down adjusted
    p-value, so ``reject <=> p_adj <= alpha``.
    """
    finite = {k: float(v) for k, v in pvalues.items() if np.isfinite(v)}
    m = len(finite)
    adjusted: dict[str, float] = {k: float("nan") for k in pvalues}
    if m == 0:
        return {"m": 0, "alpha": float(alpha), "adjusted": adjusted,
                "reject": {k: False for k in pvalues}, "n_reject": 0}

    running = 0.0
    for i, (name, p) in enumerate(sorted(finite.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - i) * p))     # monotone step-down
        adjusted[name] = running
    reject = {k: bool(np.isfinite(adjusted[k]) and adjusted[k] <= alpha) for k in pvalues}
    return {"m": m, "alpha": float(alpha), "adjusted": adjusted,
            "reject": reject, "n_reject": int(sum(reject.values()))}


def paired_effect_size(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired mean difference and Cohen's d_z for ``a - b`` (a = reference/best).

    Reported alongside every p-value: at six seeds a p-value carries almost no
    information about magnitude, and the honest question for a marketing reader is
    "how much lift is lost", not "is the loss detectable".
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    return {"mean_diff": float(np.mean(d)), "sd_diff": sd, "n": int(len(d)),
            "cohens_dz": (float(np.mean(d) / sd) if sd > 1e-12 else float("nan"))}


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CI for a test-set metric, resampling TEST ROWS.

    Complements — does not replace — the multi-seed spread (D-038). The two
    quantify different things and the manuscript must not conflate them:

    * mean +/- std over seeds  = variability of the *training procedure*
      (initialization, and on Dataset A the redrawn split);
    * this interval             = variability of the *test sample*, i.e. how much of
      the reported number is an accident of which users landed in the test split.

    A paired-across-seeds test says nothing about the second. On Dataset A the test
    split is 448 rows at a 15% positive rate (~67 positives), where PR-AUC is
    genuinely unstable, so the interval is the number a reviewer will want.

    Deterministic given ``seed``. Resamples are drawn once and reused; draws in which
    ``y_true`` has a single class are skipped (metric undefined) and counted.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals, skipped = [], 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            skipped += 1
            continue
        vals.append(float(metric_fn(yt, y_score[idx])))
    if not vals:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_boot": 0, "skipped": skipped, "alpha": float(alpha)}
    v = np.asarray(vals)
    return {"point": float(metric_fn(y_true, y_score)),
            "lo": float(np.quantile(v, alpha / 2)),
            "hi": float(np.quantile(v, 1 - alpha / 2)),
            "n_boot": len(vals), "skipped": skipped, "alpha": float(alpha)}


# backward-compatible alias
paired_wilcoxon_vs_best = significance_vs_best
