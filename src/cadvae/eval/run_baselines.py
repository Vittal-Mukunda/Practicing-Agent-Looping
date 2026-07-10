"""Phase 3 baseline runner — multi-seed frozen-representation lift on one dataset.

Fixes the "bar to beat" before any CA-DVAE work (Phase 3 gate). Writes per-seed
records + aggregated summary + Wilcoxon to ``results/`` via RunLogger.

    python -m cadvae.eval.run_baselines data=personality
    python -m cadvae.eval.run_baselines data=ecommerce eval.train_subsample=200000
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from cadvae.data import constructs as K
from cadvae.data import ecommerce as E
from cadvae.data import personality as P
from cadvae.eval.protocol import evaluate_baselines
from cadvae.eval.stats import aggregate, significance_vs_best
from cadvae.utils.run_logging import RunLogger


def _prepare(cfg: DictConfig, seed: int):
    d = cfg.data
    if d.name == "personality":
        return P.prepare(d, seed), K.constructs_personality(d, seed)
    if d.name == "ecommerce":
        return E.prepare(d, seed), K.constructs_ecommerce(d, seed)
    raise ValueError(f"unknown dataset {d.name}")


def _tasks(ps, cfg: DictConfig) -> dict:
    """Extra downstream tasks from the prepared split (D-016/D-023: B gets churned +
    next_category on the same frozen representation). Empty for Dataset A."""
    out: dict = {}
    for name, t in ps.labels.items():
        spec = {"type": t["type"], "train": t["train"], "test": t["test"]}
        if t["type"] == "multiclass":
            top_k = cfg.data.get("labels", {}).get("next_category", {}).get("top_k")
            spec["top_ks"] = tuple(top_k) if top_k else (1, 3, 5)
        out[name] = spec
    return out


def run_dataset(cfg: DictConfig) -> dict:
    device = cfg.device
    records: list[dict] = []
    for seed in cfg.eval.seeds:
        ps, ks = _prepare(cfg, int(seed))
        records.extend(evaluate_baselines(ps, ks, cfg, int(seed), device=device,
                                          tasks=_tasks(ps, cfg)))
    summary = aggregate(records)
    sig = significance_vs_best(records, metric=cfg.eval.primary_metric, head="gbt")
    return {"records": records, "summary": summary, "significance": sig}


def _print_bar(cfg: DictConfig, out: dict) -> None:
    pm = cfg.eval.primary_metric
    prim = [r for r in out["summary"] if r["task"] == "primary"]
    rows = sorted(prim, key=lambda r: (r["head"], -r[f"{pm}_mean"]))
    print(f"\n=== {cfg.data.name}: baseline downstream lift ({pm}, mean+/-std over "
          f"{len(cfg.eval.seeds)} seeds, task=primary) ===")
    for r in rows:
        print(f"  {r['head']:6s} {r['method']:12s} "
              f"roc={r['roc_auc_mean']:.4f}+/-{r['roc_auc_std']:.4f}  "
              f"pr={r['pr_auc_mean']:.4f}+/-{r['pr_auc_std']:.4f}")
    extra = sorted({r["task"] for r in out["summary"]} - {"primary"})
    for task in extra:
        print(f"  --- task={task} (gbt) ---")
        trs = [r for r in out["summary"] if r["task"] == task and r["head"] == "gbt"]
        for r in sorted(trs, key=lambda r: r["method"]):
            mets = [k for k in r if k.endswith("_mean")]
            body = "  ".join(f"{k[:-5]}={r[k]:.4f}" for k in sorted(mets))
            print(f"    {r['method']:12s} {body}")
    w = out["significance"]
    print(f"BAR TO BEAT ({pm}, gbt head, raw excluded): {w['best_method']} = {w['best_mean']:.4f}")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out = run_dataset(cfg)
    logger = RunLogger(Path(cfg.results_dir) / "phase3" / f"{cfg.data.name}_baselines", cfg)
    (logger.run_dir / "records.json").write_text(json.dumps(out["records"], indent=2))
    (logger.run_dir / "summary.json").write_text(json.dumps(out["summary"], indent=2))
    (logger.run_dir / "significance.json").write_text(json.dumps(out["significance"], indent=2))
    OmegaConf.save(cfg, logger.run_dir / "resolved_config.yaml", resolve=True)
    _print_bar(cfg, out)


if __name__ == "__main__":
    main()
