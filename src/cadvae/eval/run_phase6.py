"""Phase 6 — analysis + publication artifacts from the Phase-5 sweep (D-030/031/032).

Consumes ONLY logged evidence (Phase-5 run JSONs + saved model states, Phase-3
baseline records); trains nothing. Emits, per dataset, under
``results/phase6/{data}_analysis/``:

* ``analysis.json``  — aggregated grid, sweet spot + selection rule, paired
  significance vs the Phase-3 bar (both test families), stability numbers,
  ablation rows, and any warnings (e.g. skipped configs from a partial sweep).
* ``figures/``       — trade-off curves (MIG and alignment-R^2 variants),
  cross-seed stability bars, persona cards for the sweet-spot model.
* ``tables/``        — ablation/comparison markdown table.

Stability protocol (D-030): a fixed, seed-INDEPENDENT sample of the user universe
is embedded by each seed's model (each with its own train-only scaler — the full
per-seed pipeline is what must be stable), K-means-partitioned (K = eval.n_clusters,
single-threaded per D-028), and scored by pairwise ARI/NMI across seeds plus
bootstrap persistence on one seed. Configs compared: the D-031 sweet spot and the
two D-022 ablation points nearest to it.

    python -m cadvae.eval.run_phase6 data=personality
    python -m cadvae.eval.run_phase6 data=ecommerce eval.train_subsample=200000
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from cadvae.eval.metrics import clustering_metrics
from cadvae.eval.run_baselines import _prepare
from cadvae.eval.run_cadvae_sweep import _run_tag
from cadvae.eval.stability import (
    bootstrap_persistence,
    kmeans_assign,
    load_model_state,
    pairwise_stability,
    rebuild_cadvae,
    stability_sample,
    universe_matrix,
)
from cadvae.eval.sweep_analysis import (
    aggregate_sweep,
    best_baseline_for_task,
    load_sweep_records,
    paired_tests,
    per_seed_values,
    select_sweet_spot,
)
from cadvae.models.cadvae import encode
from cadvae.utils.run_logging import RunLogger
from cadvae.viz.figures import markdown_table, persona_cards, stability_bars, tradeoff_curve


def _grid_point(agg: list[dict], beta: float, lam: float) -> dict | None:
    for p in agg:
        if float(p["beta"]) == float(beta) and float(p["lambda_align"]) == float(lam):
            return p
    return None


def _best(points: list[dict], y: str) -> dict | None:
    finite = [p for p in points if np.isfinite(p.get(y, np.nan))]
    return max(finite, key=lambda p: p[y]) if finite else None


def ablation_points(agg: list[dict], sweet: dict, y: str) -> dict[str, dict | None]:
    """The D-022 config points reported next to the sweet spot: pure beta-VAE =
    best lambda=0 point; VAE+align = best beta=1, lambda>0 point."""
    return {
        "cadvae_sweet": sweet,
        "beta_vae": _best([p for p in agg if p["lambda_align"] == 0.0], y),
        "vae_align": _best([p for p in agg if p["beta"] == 1.0 and p["lambda_align"] > 0.0], y),
    }


def _axis_labels_from_record(rec: dict) -> dict[int, str]:
    """dim -> 'construct (R2=x)' from a sweep record's axis_alignment (best per dim)."""
    labels: dict[int, tuple[float, str]] = {}
    for name, a in rec["interpretability"]["axis_alignment"].items():
        d, r2 = int(a["best_dim"]), float(a["best_r2"])
        if d not in labels or r2 > labels[d][0]:
            labels[d] = (r2, f"{name} (R2={r2:.2f})")
    return {d: s for d, (_, s) in labels.items()}


def _stability_for_config(cfg: DictConfig, models_dir: Path, beta: float, lam: float,
                          warnings: list[str],
                          universes: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict | None:
    """Cross-seed ARI/NMI + bootstrap persistence + clustering quality for one grid
    point (D-030). ``universes`` memoizes per-seed (uid, X) across configs — the
    prepare() reload is the dominant Phase-6 cost on Dataset B."""
    seeds = [int(s) for s in cfg.eval.seeds]
    missing = [f"{_run_tag(s, beta, lam)}.pt" for s in seeds
               if not (models_dir / f"{_run_tag(s, beta, lam)}.pt").exists()]
    if missing:                                # pre-check BEFORE any embedding work
        warnings.append(f"stability: config (beta={beta:g}, lambda={lam:g}) skipped — "
                        f"missing models: {missing}")
        return None
    k = int(cfg.eval.n_clusters)
    cap = int(cfg.phase6.stability_users)
    sample_seed = int(cfg.phase6.stability_seed)
    assignments: dict[int, np.ndarray] = {}
    ref_uid = None
    boot = None
    clu: list[dict] = []
    for seed in seeds:
        model = rebuild_cadvae(load_model_state(models_dir / f"{_run_tag(seed, beta, lam)}.pt"))
        if seed not in universes:
            ps, _ = _prepare(cfg, seed)
            universes[seed] = universe_matrix(ps)
        uid, X = universes[seed]
        if ref_uid is None:
            ref_uid = uid
            idx = stability_sample(len(uid), cap, sample_seed)
        elif not np.array_equal(uid, ref_uid):
            raise AssertionError("user universes differ across seeds — split logic broken")
        Z = encode(model, X[idx]).astype(np.float64)
        assignments[seed] = kmeans_assign(Z, k, seed)
        # clustering quality on the SAME embedding/partition (CLAUDE.md protocol)
        clu.append(clustering_metrics(Z, assignments[seed], seed=seed))
        if boot is None:                       # bootstrap on the first seed's embedding
            boot = bootstrap_persistence(Z, k, n_boot=int(cfg.phase6.n_boot), seed=seed)
    out = pairwise_stability(assignments)
    out["bootstrap"] = boot
    out["n_users"] = int(len(next(iter(assignments.values()))))
    out["clustering"] = {
        m: float(np.nanmean([c[m] for c in clu]))
        for m in ("silhouette", "davies_bouldin", "calinski_harabasz")}
    return out


def analyze(cfg: DictConfig) -> dict:
    name = cfg.data.name
    head = str(cfg.phase6.head)
    pm = str(cfg.eval.primary_metric)
    y = f"{pm}_mean"
    sweep_dir = Path(cfg.results_dir) / "phase5" / f"{name}_sweep"
    phase3_dir = Path(cfg.results_dir) / "phase3" / f"{name}_baselines"
    out_dir = Path(cfg.results_dir) / "phase6" / f"{name}_analysis"
    for p, phase in ((sweep_dir, "5 (sweep)"), (phase3_dir, "3 (baselines)")):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run Phase {phase} first")
    logger = RunLogger(out_dir, cfg)
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    warnings: list[str] = []

    records = load_sweep_records(sweep_dir)
    agg = aggregate_sweep(records, head=head)

    sig3 = json.loads((phase3_dir / "significance.json").read_text())
    bar = {"name": sig3["best_method"], "value": float(sig3["best_mean"])}
    summary3 = json.loads((phase3_dir / "summary.json").read_text())
    raw_rows = [r for r in summary3 if r["method"] == "raw" and r["head"] == head
                and r.get("task", "primary") == "primary"]
    ceiling = {"name": "raw", "value": float(raw_rows[0][y])} if raw_rows else None

    # model selection on VALIDATION metrics, reporting on test (D-033): selecting
    # (beta, lambda) by the test metric would be tuning on test
    y_sel = f"{cfg.phase6.get('select_on', 'val_' + pm)}_mean"
    if not any(np.isfinite(p.get(y_sel, np.nan)) for p in agg):
        warnings.append(f"selection metric '{y_sel}' absent from sweep records — "
                        f"falling back to TEST metric '{y}' (pre-D-033 sweep?)")
        y_sel = y
    sweet_sel = select_sweet_spot(agg, x=str(cfg.phase6.x_metric), y=y_sel,
                                  y_slack=cfg.phase6.get("y_slack"))
    sweet = sweet_sel["point"]
    sb, sl = float(sweet["beta"]), float(sweet["lambda_align"])

    # paired significance per task: sweet-spot CA-DVAE (TEST metrics) vs the best
    # PER-TASK baseline — beating only the primary-bar method on a task where a
    # different baseline is stronger would be a straw-man comparison
    significance: dict[str, dict] = {}
    task_specs = [("primary", pm, pm),
                  ("churned", "pr_auc", "churned_pr_auc"),
                  ("next_category", "acc_at_1", "next_category_acc_at_1"),
                  ("next_category", "macro_ovr_auc", "next_category_macro_ovr_auc")]
    for task, base_metric, flat_metric in task_specs:
        try:
            a = per_seed_values(records, sb, sl, metric=flat_metric, head=head)
        except KeyError:
            continue                               # dataset without this task (A)
        bb = best_baseline_for_task(phase3_dir / "records.json", task=task,
                                    metric=base_metric, head=head)
        if not a or bb is None:
            continue
        key = "primary" if task == "primary" else f"{task}/{base_metric}"
        significance[key] = {"task": task, "metric": base_metric,
                             "baseline": bb["method"],
                             **paired_tests(a, bb["by_seed"])}

    # figures + tables
    files: dict[str, str] = {}
    files["tradeoff_mig"] = str(tradeoff_curve(
        agg, fig_dir / "tradeoff_mig.png", x="mig_mean", y=y, bar=bar,
        ceiling=ceiling, sweet=sweet, title=f"{name}: downstream vs MIG ({head})"))
    try:
        files["tradeoff_r2"] = str(tradeoff_curve(
            agg, fig_dir / "tradeoff_r2.png", x="r2_mean_mean", y=y, bar=bar,
            ceiling=ceiling, sweet=sweet,
            title=f"{name}: downstream vs alignment R2 ({head})"))
    except ValueError as e:                        # e.g. only lambda=0 runs exist
        warnings.append(f"tradeoff_r2 skipped: {e}")

    # multi-task comparison table — the blessed multi-task claim, one row per task
    mt_rows = []
    for sig in significance.values():
        cad = sig["mean_a"]
        mt_rows.append({"task": sig["task"], "metric": sig["metric"],
                        "best_baseline": sig["baseline"], "baseline_mean": sig["mean_b"],
                        "cadvae_mean": cad, "delta": cad - sig["mean_b"],
                        "wilcoxon_p": sig.get("wilcoxon_p", float("nan")),
                        "ttest_p": sig.get("ttest_p", float("nan"))})
    if mt_rows:
        files["multitask_table"] = str(markdown_table(
            mt_rows, tab_dir / "multitask.md",
            title=f"{name}: CA-DVAE sweet spot vs best PER-TASK baseline "
                  f"({head} head, test set, paired over seeds)"))

    abl = ablation_points(agg, sweet, y_sel)       # same selection metric as the sweet spot
    abl_rows = []
    for label, pt in abl.items():
        if pt is None:
            warnings.append(f"ablation point '{label}' absent from grid")
            continue
        abl_rows.append({"config": label, "beta": pt["beta"], "lambda": pt["lambda_align"],
                         pm: pt[y], f"{pm}_std": pt.get(f"{pm}_std", float("nan")),
                         "mig": pt.get("mig_mean", float("nan")),
                         "r2": pt.get("r2_mean_mean", float("nan"))})
    abl_rows.append({"config": f"bar ({bar['name']})", "beta": "-", "lambda": "-",
                     pm: bar["value"], f"{pm}_std": float("nan"),
                     "mig": float("nan"), "r2": float("nan")})
    files["ablation_table"] = str(markdown_table(
        abl_rows, tab_dir / "ablations.md",
        title=f"{name}: ablations vs bar ({pm}, {head}, mean over seeds)"))

    stability: dict[str, dict] = {}
    models_dir = sweep_dir / "models"
    universes: dict[int, tuple[np.ndarray, np.ndarray]] = {}   # per-seed cache
    for label, pt in abl.items():
        if pt is None:
            continue
        stab = _stability_for_config(cfg, models_dir, float(pt["beta"]),
                                     float(pt["lambda_align"]), warnings, universes)
        if stab is not None:
            stability[label] = stab
    if stability:
        files["stability"] = str(stability_bars(
            {k: v for k, v in stability.items()}, fig_dir / "stability.png",
            title=f"{name}: persona stability (K={cfg.eval.n_clusters})"))
    else:
        warnings.append("no stability results (missing sweep models?)")

    # persona cards from the sweet-spot model at the lowest seed
    card_seed = min(int(s) for s in cfg.eval.seeds)
    card_model = models_dir / f"{_run_tag(card_seed, sb, sl)}.pt"
    rec = next((r for r in records
                if int(r["seed"]) == card_seed and float(r["beta"]) == sb
                and float(r["lambda_align"]) == sl), None)
    if card_model.exists() and rec is not None:
        ps, _ = _prepare(cfg, card_seed)
        try:
            files["persona_cards"] = str(persona_cards(
                load_model_state(card_model), ps.fitted["scaler"], ps.feature_names,
                fig_dir / "persona_cards.png",
                axis_labels=_axis_labels_from_record(rec),
                span=float(cfg.phase6.traversal_span),
                top_n=int(cfg.phase6.card_top_features)))
        except ValueError as e:                    # sweet spot at lambda=0 (no aligned dims)
            warnings.append(f"persona cards skipped: {e}")
    else:
        warnings.append(f"persona cards skipped: {card_model.name} or its record missing")

    analysis = {"dataset": name, "head": head, "primary_metric": pm,
                "bar": bar, "ceiling": ceiling,
                "sweet_spot": sweet_sel, "significance_vs_bar": significance,
                "ablations": abl_rows, "stability": stability,
                "grid_aggregate": agg, "files": files, "warnings": warnings}
    (logger.run_dir / "analysis.json").write_text(json.dumps(analysis, indent=1))
    print(f"PHASE 6 ANALYSIS COMPLETE -> {out_dir}", flush=True)
    for w in warnings:
        print(f"  WARN: {w}", flush=True)
    return analysis


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    analyze(cfg)


if __name__ == "__main__":
    main()
