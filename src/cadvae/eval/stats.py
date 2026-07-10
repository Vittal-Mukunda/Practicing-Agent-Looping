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
    wilcox, tt = {}, {}
    n_used = 0
    for m in methods:
        if m == best:
            continue
        mb = _by_seed(records, m, head, metric, task)
        seeds = sorted(set(best_by_seed) & set(mb))
        n_used = len(seeds)
        a = np.array([best_by_seed[s] for s in seeds])
        b = np.array([mb[s] for s in seeds])
        if len(seeds) < 2 or np.allclose(a, b):
            wilcox[m] = tt[m] = float("nan")
        else:
            wilcox[m] = float(wilcoxon(a, b).pvalue)
            tt[m] = float(ttest_rel(a, b).pvalue)
    return {"best_method": best, "best_mean": float(means[best]), "metric": metric,
            "head": head, "task": task, "n_seeds": n_used,
            "excluded_from_best": list(exclude_from_best),
            "wilcoxon_floor": (2.0 ** -(n_used - 1) if n_used >= 2 else float("nan")),
            "wilcoxon_p": wilcox, "ttest_p": tt}


# backward-compatible alias
paired_wilcoxon_vs_best = significance_vs_best
