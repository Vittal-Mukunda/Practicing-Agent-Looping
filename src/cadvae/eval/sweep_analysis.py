"""Phase-6 analysis of the Phase-5 sweep (D-031).

Pure functions from sweep records (the per-run JSONs written by
``run_cadvae_sweep``) to the paper's tables and curve data:

* ``flatten_record`` / ``aggregate_sweep`` — per-(beta, lambda) mean +/- std of
  every numeric metric across seeds (NaN-aware: the lambda=0 ablation has no
  alignment R^2; Dataset B has NaN interpretability factors by design, D-026).
* ``pareto_frontier`` — non-dominated points in the (interpretability, downstream)
  plane; the trade-off curve's efficient set.
* ``select_sweet_spot`` — D-031 rule: among frontier points whose downstream mean
  is within ``y_slack`` of the best downstream mean anywhere on the grid (default
  slack = the best point's own seed-std, i.e. "statistically indistinguishable from
  the best"), pick the one with the highest interpretability. Falls back to the
  best-downstream point if the frontier is empty under the slack. The rule is a
  *reported, falsifiable choice* — both inputs and the chosen point are logged.
* ``paired_tests`` — paired Wilcoxon + t (both families, CLAUDE.md statistics rule)
  between two per-seed metric dicts, e.g. sweet-spot CA-DVAE vs the Phase-3 bar
  method. Wilcoxon degenerates when all per-seed differences are zero; that is
  reported as p=1.0 (no evidence of difference) instead of crashing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

_GRID_KEYS = ("beta", "lambda_align")
_ID_KEYS = (*_GRID_KEYS, "seed")


def load_sweep_records(sweep_dir: str | Path) -> list[dict]:
    runs_dir = Path(sweep_dir) / "runs"
    files = sorted(runs_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"no sweep records in {runs_dir} — run Phase 5 first "
            f"(python -m cadvae.eval.run_cadvae_sweep data=...)")
    out = []
    for f in files:
        try:
            out.append(json.loads(f.read_text()))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"corrupt sweep record {f}: {e} — delete that file and relaunch the "
                f"sweep; resume will regenerate just that run") from e
    return out


def flatten_record(rec: dict, head: str = "gbt") -> dict:
    """One sweep record -> flat numeric row (primary metrics unprefixed, extra
    tasks prefixed with their name, interpretability + alignment + kl_free)."""
    row: dict = {k: rec[k] for k in _ID_KEYS}
    row["aligned_dims"] = rec["aligned_dims"]
    for task, heads in rec["downstream"].items():
        prefix = "" if task == "primary" else f"{task}_"
        for m, v in heads[head].items():
            if isinstance(v, (int, float)):
                row[f"{prefix}{m}"] = float(v)
    # validation metrics for model selection (D-033); absent in pre-D-033 records
    for m, v in rec.get("downstream_val", {}).get("primary", {}).get(head, {}).items():
        if isinstance(v, (int, float)):
            row[f"val_{m}"] = float(v)
    row["mig"] = float(rec["interpretability"]["mig"])
    row["sap"] = float(rec["interpretability"]["sap"])
    align = rec.get("alignment")
    row["r2_mean"] = float(align["r2_mean"]) if align else float("nan")
    row["r2_rfm_mean"] = float(align["r2_rfm_mean"]) if align else float("nan")
    row["kl_free"] = float(rec["loss_final"]["kl_free"])
    return row


def aggregate_sweep(records: list[dict], head: str = "gbt") -> list[dict]:
    """Group flat rows by (beta, lambda_align) -> NaN-aware mean/std per metric."""
    rows = [flatten_record(r, head) for r in records]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["beta"], r["lambda_align"])].append(r)
    out = []
    for (beta, lam), rs in sorted(groups.items()):
        agg = {"beta": beta, "lambda_align": lam, "n_seeds": len(rs)}
        metrics = [k for k in rs[0] if k not in _ID_KEYS]
        for m in [k for k in metrics if all(k in r for r in rs)]:
            vals = np.array([r[m] for r in rs], dtype=float)
            agg[f"{m}_mean"] = float(np.nanmean(vals)) if not np.isnan(vals).all() \
                else float("nan")
            agg[f"{m}_std"] = float(np.nanstd(vals, ddof=1)) \
                if np.isfinite(vals).sum() > 1 else 0.0
        out.append(agg)
    return out


def pareto_frontier(points: list[dict], x: str, y: str) -> list[dict]:
    """Non-dominated points maximizing both x and y (NaN points excluded)."""
    finite = [p for p in points if np.isfinite(p.get(x, np.nan))
              and np.isfinite(p.get(y, np.nan))]
    front = []
    for p in finite:
        dominated = any(
            q[x] >= p[x] and q[y] >= p[y] and (q[x] > p[x] or q[y] > p[y])
            for q in finite if q is not p)
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p[x])


def select_sweet_spot(agg: list[dict], x: str = "mig_mean", y: str = "pr_auc_mean",
                      y_slack: float | None = None) -> dict:
    """D-031 sweet-spot rule (see module docstring). Returns point + rule inputs."""
    finite_y = [p for p in agg if np.isfinite(p.get(y, np.nan))]
    if not finite_y:
        raise ValueError(f"no grid point has a finite '{y}'")
    best = max(finite_y, key=lambda p: p[y])
    slack = float(y_slack) if y_slack is not None \
        else float(best.get(y.replace("_mean", "_std"), 0.0))
    front = pareto_frontier(agg, x, y)
    eligible = [p for p in front if p[y] >= best[y] - slack]
    chosen = max(eligible, key=lambda p: p[x]) if eligible else best
    return {"point": chosen, "rule": {"x": x, "y": y, "y_slack": slack,
                                      "best_y": {k: best[k] for k in ("beta", "lambda_align", y)},
                                      "frontier_size": len(front),
                                      "fallback_used": not eligible}}


def per_seed_values(records: list[dict], beta: float, lam: float,
                    metric: str = "pr_auc", head: str = "gbt") -> dict[int, float]:
    """seed -> flat metric value at one grid point."""
    out = {}
    for rec in records:
        if float(rec["beta"]) == float(beta) and float(rec["lambda_align"]) == float(lam):
            out[int(rec["seed"])] = flatten_record(rec, head)[metric]
    return out


def baseline_per_seed(records_path: str | Path, method: str, metric: str = "pr_auc",
                      head: str = "gbt", task: str = "primary") -> dict[int, float]:
    """seed -> metric for one baseline method from a Phase-3 records.json."""
    recs = json.loads(Path(records_path).read_text())
    return {int(r["seed"]): float(r[metric]) for r in recs
            if r["method"] == method and r["head"] == head
            and r.get("task", "primary") == task and metric in r}


def best_baseline_for_task(records_path: str | Path, task: str, metric: str,
                           head: str = "gbt",
                           exclude: tuple[str, ...] = ("raw",)) -> dict | None:
    """Strongest baseline for ONE task by mean metric (raw excluded, D-024).

    The multi-task claim must beat the best PER-TASK incumbent, not just the
    primary-bar method — on next-category the strongest baseline differs from the
    primary bar, and comparing against the wrong one is a straw man."""
    recs = json.loads(Path(records_path).read_text())
    sel = [r for r in recs if r["head"] == head and r.get("task", "primary") == task
           and metric in r and r["method"] not in exclude]
    if not sel:
        return None
    means: dict[str, list[float]] = defaultdict(list)
    for r in sel:
        means[r["method"]].append(float(r[metric]))
    best = max(means, key=lambda m: float(np.mean(means[m])))
    return {"method": best, "mean": float(np.mean(means[best])),
            "by_seed": {int(r["seed"]): float(r[metric]) for r in sel
                        if r["method"] == best}}


def paired_tests(a_by_seed: dict[int, float], b_by_seed: dict[int, float]) -> dict:
    """Paired Wilcoxon + t over shared seeds (a vs b). NaN p-values when n < 2."""
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    a = np.array([a_by_seed[s] for s in seeds])
    b = np.array([b_by_seed[s] for s in seeds])
    out = {"n_seeds": len(seeds), "seeds": seeds,
           "mean_a": float(a.mean()) if len(a) else float("nan"),
           "mean_b": float(b.mean()) if len(b) else float("nan"),
           "wilcoxon_floor": (2.0 ** -(len(seeds) - 1) if len(seeds) >= 2 else float("nan"))}
    if len(seeds) < 2:
        out["wilcoxon_p"] = out["ttest_p"] = float("nan")
        return out
    if np.allclose(a, b):
        # all differences ~zero: no evidence of any difference; wilcoxon() raises here
        out["wilcoxon_p"] = out["ttest_p"] = 1.0
        return out
    out["wilcoxon_p"] = float(wilcoxon(a, b).pvalue)
    out["ttest_p"] = float(ttest_rel(a, b).pvalue)
    return out
