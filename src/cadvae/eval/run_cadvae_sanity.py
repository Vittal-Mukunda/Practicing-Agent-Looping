"""Phase 4 CA-DVAE **sanity run** — NOT the Phase-5 sweep.

One dataset, one seed, one (beta, lambda_align) point. It (1) trains the CA-DVAE and
logs the per-epoch loss curves, (2) freezes it and runs the *same* frozen-representation
heads as the Phase-3 baselines (L2-logistic + HistGBT) for a downstream-lift sanity
check against the fixed bar, and (3) reports alignment fidelity (per-construct R^2 of
the linear alignment head on held-out test).

Purpose = gate evidence that the model trains, the losses behave, alignment actually
works, and the end-to-end pipeline is sound — BEFORE committing GPU to the sweep. A
single arbitrary (beta, lambda_align) point is **not** expected to beat the bar; the
trade-off is explored in Phase 5. We never tune this to beat the bar (integrity §3).

    python -m cadvae.eval.run_cadvae_sanity data=personality
    python -m cadvae.eval.run_cadvae_sanity data=ecommerce eval.train_subsample=50000 \
        model.max_epochs=40
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import r2_score

from cadvae.eval.protocol import _fit_eval_heads, _subsample
from cadvae.eval.run_baselines import _prepare
from cadvae.models.cadvae import align_predict, encode, resolve_aligned_dims, train_cadvae
from cadvae.utils.run_logging import RunLogger


def _phase3_bar(cfg: DictConfig) -> dict | None:
    """Read the committed Phase-3 bar (best baseline) for context, if present."""
    p = Path(cfg.results_dir) / "phase3" / f"{cfg.data.name}_baselines" / "significance.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return {"method": s["best_method"], "pr_auc": s["best_mean"], "head": s.get("head", "gbt")}


def run_sanity(cfg: DictConfig, logger: RunLogger) -> dict:
    device = cfg.device
    seed = int(cfg.seed)
    ps, ks = _prepare(cfg, seed)

    # train subsample (matches the baseline protocol for a fair comparison on B)
    tr = _subsample(len(ps.y_train), cfg, seed)
    Xtr, ytr = ps.X_train[tr], ps.y_train[tr]
    Ctr = ks.C_train[tr]
    Xte, yte = ps.X_test, ps.y_test

    n_constructs = len(ks.names)
    aligned = resolve_aligned_dims(cfg.model, n_constructs)
    model, history = train_cadvae(
        Xtr, Ctr, cfg.model, seed, device=device, n_constructs=n_constructs,
        log_fn=logger.log_metrics,
    )

    # frozen downstream lift (posterior-mean embedding), identical heads to Phase 3
    Etr = encode(model, Xtr, device=device).astype(np.float64)
    Ete = encode(model, Xte, device=device).astype(np.float64)
    downstream = _fit_eval_heads(Etr, ytr, Ete, yte, seed)

    # alignment fidelity: per-construct R^2 of the linear head on held-out test
    alignment: dict = {"aligned_dims": aligned, "n_constructs": n_constructs}
    pred_te = align_predict(model, Xte, device=device)
    if pred_te is not None:
        r2 = r2_score(ks.C_test, pred_te, multioutput="raw_values")
        alignment["r2_per_construct"] = {n: float(v) for n, v in zip(ks.names, r2, strict=True)}
        alignment["r2_mean"] = float(np.mean(r2))
        alignment["r2_rfm_mean"] = float(np.mean(r2[:3]))  # R, F, M

    return {
        "dataset": cfg.data.name, "seed": seed,
        "beta": float(cfg.model.beta), "lambda_align": float(cfg.model.lambda_align),
        "aligned_dims": aligned,
        "history_final": history[-1], "downstream": downstream, "alignment": alignment,
        "phase3_bar": _phase3_bar(cfg),
    }


def _print_summary(out: dict) -> None:
    # ASCII only — the Windows console (cp1252) cannot encode beta/lambda/superscripts.
    pm = "pr_auc"
    print(f"\n=== CA-DVAE SANITY ({out['dataset']}, seed {out['seed']}, "
          f"beta={out['beta']}, lambda={out['lambda_align']}, "
          f"aligned_dims={out['aligned_dims']}) ===")
    print("  [single point -- NOT the Phase-5 sweep; not tuned to beat the bar]")
    hf = out["history_final"]
    print(f"  final losses: total={hf['total']:.3f} recon={hf['recon']:.3f} "
          f"kl_free={hf['kl_free']:.3f} kl_aligned={hf['kl_aligned']:.3f} align={hf['align']:.3f}")
    for head, m in out["downstream"].items():
        print(f"  {head:6s} roc={m['roc_auc']:.4f}  pr={m['pr_auc']:.4f}  "
              f"(base_rate={m['base_rate']:.3f})")
    bar = out["phase3_bar"]
    if bar:
        best_pr = max(m[pm] for m in out["downstream"].values())
        delta = best_pr - bar["pr_auc"]
        print(f"  Phase-3 bar: {bar['method']} {pm}={bar['pr_auc']:.4f} "
              f"-> CA-DVAE best {pm}={best_pr:.4f} (delta={delta:+.4f})")
    a = out["alignment"]
    if "r2_mean" in a:
        print(f"  alignment R2: mean={a['r2_mean']:.3f}  RFM-mean={a['r2_rfm_mean']:.3f}")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = RunLogger(Path(cfg.results_dir) / "phase4" / f"{cfg.data.name}_cadvae_sanity", cfg)
    out = run_sanity(cfg, logger)
    (logger.run_dir / "sanity.json").write_text(json.dumps(out, indent=2))
    OmegaConf.save(cfg, logger.run_dir / "resolved_config.yaml", resolve=True)
    _print_summary(out)


if __name__ == "__main__":
    main()
