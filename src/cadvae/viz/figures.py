"""Publication figures (Phase 6, D-032). Matplotlib only, headless (Agg), each
function renders one artifact to ``out_path`` and returns the path.

Conventions: PNG at 200 dpi (vector PDF can be added at camera-ready), one chart
per figure, explicit axis labels with units/head, error bars = seed std. No style
packages — reviewers reproduce with a bare matplotlib install.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures render identically with no display
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cadvae.eval.stability import rebuild_cadvae  # noqa: E402

_X_LABELS = {"mig": "MIG (construct factors)", "sap": "SAP",
             "r2_mean": "alignment R$^2$ (mean over constructs)"}


def tradeoff_curve(agg: list[dict], out_path: str | Path, x: str = "mig_mean",
                   y: str = "pr_auc_mean", bar: dict | None = None,
                   ceiling: dict | None = None, sweet: dict | None = None,
                   title: str = "") -> Path:
    """THE headline figure: downstream performance vs interpretability.

    One series per lambda (beta varies along each series); horizontal lines mark
    the Phase-3 bar and the raw-feature reference ceiling; the D-031 sweet spot is
    starred. ``agg`` rows need ``{x, y}`` means and ``*_std`` companions.
    """
    pts = [p for p in agg if np.isfinite(p.get(x, np.nan)) and np.isfinite(p.get(y, np.nan))]
    if not pts:
        raise ValueError(f"no grid point has finite '{x}' and '{y}' — nothing to plot")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for lam in sorted({p["lambda_align"] for p in pts}):
        series = sorted((p for p in pts if p["lambda_align"] == lam), key=lambda p: p["beta"])
        ax.errorbar([p[x] for p in series], [p[y] for p in series],
                    xerr=[p.get(x.replace("_mean", "_std"), 0.0) for p in series],
                    yerr=[p.get(y.replace("_mean", "_std"), 0.0) for p in series],
                    marker="o", ms=4, lw=1.2, capsize=2,
                    label=rf"$\lambda$={lam:g}")
        for p in series:
            ax.annotate(rf"$\beta$={p['beta']:g}", (p[x], p[y]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    if bar:
        ax.axhline(bar["value"], color="0.25", ls="--", lw=1)
        ax.annotate(f"bar: {bar['name']} = {bar['value']:.3f}", (0.99, bar["value"]),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=7)
    if ceiling:
        ax.axhline(ceiling["value"], color="0.6", ls=":", lw=1)
        ax.annotate(f"ceiling: {ceiling['name']} = {ceiling['value']:.3f}",
                    (0.99, ceiling["value"]), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7)
    if sweet and np.isfinite(sweet.get(x, np.nan)):
        ax.plot(sweet[x], sweet[y], marker="*", ms=16, mfc="gold", mec="black", zorder=5)
    ax.set_xlabel(_X_LABELS.get(x.replace("_mean", ""), x))
    ax.set_ylabel(y.replace("_mean", "").replace("_", " ") + " (test)")
    ax.set_title(title or "Interpretability-performance trade-off")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    return _save(fig, out_path)


def stability_bars(results: dict[str, dict], out_path: str | Path,
                   title: str = "Persona stability across seeds") -> Path:
    """Grouped bars of cross-seed ARI and NMI (mean +/- std) per configuration."""
    if not results:
        raise ValueError("no stability results to plot")
    names = list(results)
    xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(4.5, 1.6 * len(names)), 4.0))
    for off, metric, color in ((-0.18, "ari", "#3465a4"), (0.18, "nmi", "#a40000")):
        ax.bar(xs + off, [results[n][f"{metric}_mean"] for n in names], width=0.34,
               yerr=[results[n][f"{metric}_std"] for n in names], capsize=3,
               color=color, label=metric.upper())
    ax.set_xticks(xs, names, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("cross-seed agreement")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    return _save(fig, out_path)


def markdown_table(rows: list[dict], out_path: str | Path,
                   columns: list[str] | None = None, title: str = "") -> Path:
    """Rows of dicts -> a markdown table artifact (ablation/comparison tables)."""
    if not rows:
        raise ValueError("no rows for table")
    cols = columns or list(rows[0])
    lines = ([f"# {title}", ""] if title else [])
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def persona_cards(state: dict, scaler, feature_names: list[str], out_path: str | Path,
                  axis_labels: dict[int, str] | None = None,
                  dims: list[int] | None = None, span: float = 2.0,
                  top_n: int = 8) -> Path:
    """Latent-traversal persona cards (CLAUDE.md): move ONE aligned axis from
    -span to +span (prior units), decode, and show the top raw-feature changes
    (inverse-standardized), one panel per axis. ``axis_labels`` names each dim
    with its best-aligned construct (from the sweep record's axis_alignment)."""
    model = rebuild_cadvae(state)
    if dims is None:
        if state["aligned_dims"] == 0:
            raise ValueError("model has no aligned dims — pass dims= explicitly "
                             "to traverse free axes")
        dims = list(range(int(state["aligned_dims"])))
    axis_labels = axis_labels or {}
    fig, axes = plt.subplots(len(dims), 1, figsize=(7.0, 1.9 * len(dims) + 0.8),
                             squeeze=False, layout="constrained")
    with torch.no_grad():
        for row, dim in enumerate(dims):
            z = torch.zeros(2, int(state["latent_dim"]))
            z[0, dim], z[1, dim] = -span, span
            xh = model.decoder(z).numpy()
            # Inverse to raw units so the card reads as a customer, not a z-score.
            # The scaler may cover only a LEADING block of the feature vector
            # (Dataset A: 23 scaled numerics then 11 unscaled one-hots); the
            # remainder stays in decoded units (one-hot deltas = probability shifts).
            n_sc = int(scaler.n_features_in_)
            raw = xh.copy()
            raw[:, :n_sc] = scaler.inverse_transform(xh[:, :n_sc])
            delta = raw[1] - raw[0]
            # Bars are STANDARDIZED deltas (scale-free — comparable across
            # features; raw units would let A's Income, ~1e4, dwarf every bar),
            # annotated with the raw-unit change so the card still reads as a
            # customer. Ranking likewise on standardized magnitude.
            zd = xh[1] - xh[0]
            top = np.argsort(np.abs(zd))[::-1][:top_n]
            vals = zd[top][::-1]
            raws = delta[top][::-1]
            ax = axes[row, 0]
            ax.barh(range(len(top)), vals,
                    color=["#3465a4" if d > 0 else "#a40000" for d in vals])
            for i, (v, rv) in enumerate(zip(vals, raws, strict=True)):
                ax.annotate(f"{rv:+,.3g}", xy=(v, i),
                            xytext=(3 if v >= 0 else -3, 0),
                            textcoords="offset points",
                            ha="left" if v >= 0 else "right", va="center",
                            fontsize=6, color="0.25")
            ax.set_yticks(range(len(top)), [feature_names[i] for i in top][::-1], fontsize=7)
            ax.axvline(0, color="0.3", lw=0.8)
            ax.margins(x=0.15)                     # room for the annotations
            label = axis_labels.get(dim, "unaligned")
            ax.set_title(f"axis z[{dim}] — {label}  (decode at ±{span:g})", fontsize=9)
            ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Persona cards: per-axis latent traversals "
                 "(bars: standardized Δ; labels: raw-unit Δ)")
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
