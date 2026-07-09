"""Aggregate multi-seed baseline records → mean ± std + paired significance.

CLAUDE.md statistical-rigor rule: every headline number is mean ± std over ≥5 seeds,
plus a paired Wilcoxon vs the strongest baseline (never a single run).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

METRICS = ["roc_auc", "pr_auc", "prec_at_10pct"]


def aggregate(records: list[dict]) -> list[dict]:
    """Mean ± std of each metric per (method, head) across seeds."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["method"], r["head"])].append(r)
    out = []
    for (method, head), rs in groups.items():
        row = {"method": method, "head": head, "n_seeds": len(rs)}
        for m in METRICS:
            vals = np.array([r[m] for r in rs if m in r], dtype=float)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def _by_seed(records, method, head, metric):
    d = {r["seed"]: r[metric] for r in records if r["method"] == method and r["head"] == head}
    return d


def significance_vs_best(records: list[dict], metric: str = "pr_auc",
                         head: str = "gbt") -> dict:
    """Paired significance of every method's per-seed metric vs the best-mean method.

    Reports BOTH a paired Wilcoxon signed-rank and a paired t-test (CLAUDE.md allows
    either). NOTE: two-sided Wilcoxon has a discrete p-floor of 2^-(n-1) — at n=5
    seeds that floor is 0.0625, so it can never reach p<0.05 no matter how consistent
    the win. The paired t-test can, and ``n_seeds`` is reported so the reader can
    judge. For headline claims, ≥6 seeds are recommended (Wilcoxon floor → 0.03125).
    """
    methods = sorted({r["method"] for r in records if r["head"] == head})
    means = {m: np.mean([r[metric] for r in records
                         if r["method"] == m and r["head"] == head]) for m in methods}
    best = max(means, key=lambda m: means[m])
    best_by_seed = _by_seed(records, best, head, metric)
    wilcox, tt = {}, {}
    n_used = 0
    for m in methods:
        if m == best:
            continue
        mb = _by_seed(records, m, head, metric)
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
            "head": head, "n_seeds": n_used,
            "wilcoxon_floor": (2.0 ** -(n_used - 1) if n_used >= 2 else float("nan")),
            "wilcoxon_p": wilcox, "ttest_p": tt}


# backward-compatible alias
paired_wilcoxon_vs_best = significance_vs_best
