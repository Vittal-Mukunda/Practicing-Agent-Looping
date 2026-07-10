"""Phase 5 — CA-DVAE multi-seed x (beta, lambda_align) sweep (D-029).

Generates the paper's headline data: for every grid point and seed, the frozen-rep
downstream metrics (primary + extra tasks, same heads as Phase 3), the alignment
R^2, and the interpretability scores (MIG / SAP / axis alignment) whose trade-off
against downstream lift is the headline curve.

Design (D-029, deviations from a naive Hydra multirun, all deliberate):
* **Internal grid loop, one process per dataset.** ``prepare()`` (parquet load,
  split, constructs) is paid once per SEED and reused across all grid points —
  on Dataset B that saves ~24x the most expensive non-GPU step.
* **Resume granularity = one (seed, beta, lambda) run.** Each run writes
  ``runs/s{seed}_b{beta}_l{lambda}.json`` on completion; existing files are
  skipped on relaunch, so a mid-night crash costs one run, not the batch.
  Delete the sweep dir if the PROTOCOL changes (config hash is logged).
* **Model state_dicts are saved** (``models/*.pt``, ~120 KB each) so Phase 6 can
  compute clustering/stability (ARI/NMI, bootstrap) for selected configs without
  retraining; per-run KMeans is deliberately NOT run here (it is the single most
  expensive CPU step under the D-028 single-thread policy and is only needed for
  the configs Phase 6 actually inspects).
* The lambda=0 column doubles as the pure beta-VAE ablation (aligned_dims forced
  0, D-022); beta=1 row doubles as the VAE+align ablation.

    python -m cadvae.eval.run_cadvae_sweep data=personality
    python -m cadvae.eval.run_cadvae_sweep data=ecommerce \
        eval.train_subsample=200000 model.max_epochs=40
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import r2_score

from cadvae.eval.interpretability import interpretability_summary
from cadvae.eval.protocol import _fit_eval_heads, _fit_eval_heads_multiclass, _subsample
from cadvae.eval.run_baselines import _prepare, _tasks
from cadvae.models.cadvae import align_predict, encode, resolve_aligned_dims, train_cadvae
from cadvae.utils.run_logging import RunLogger


def _run_tag(seed: int, beta: float, lam: float) -> str:
    return f"s{seed}_b{beta:g}_l{lam:g}"


def _task_heads(R_tr, R_te, tasks: dict, tr: np.ndarray, seed: int) -> dict:
    """Extra-task heads on the frozen representation (mirrors protocol.emit)."""
    out: dict = {}
    for tname, t in tasks.items():
        ytr_t, mtr = t["train"]
        yte_t, mte = t["test"]
        mtr, ytr_t = mtr[tr], ytr_t[tr]
        if t["type"] == "multiclass":
            out[tname] = _fit_eval_heads_multiclass(
                R_tr[mtr], ytr_t[mtr], R_te[mte], yte_t[mte], seed,
                top_ks=tuple(t.get("top_ks", (1, 3, 5))))
        else:
            out[tname] = _fit_eval_heads(R_tr[mtr], ytr_t[mtr], R_te[mte], yte_t[mte], seed)
    return out


def run_point(ps, ks, tasks: dict, cfg: DictConfig, seed: int,
              beta: float, lam: float, device: str = "cuda") -> tuple[dict, dict]:
    """One (seed, beta, lambda) run -> (record, model state for Phase 6)."""
    t0 = time.time()
    model_cfg = OmegaConf.merge(cfg.model, {"beta": beta, "lambda_align": lam})
    n_constructs = len(ks.names)
    aligned = resolve_aligned_dims(model_cfg, n_constructs)

    tr = _subsample(len(ps.y_train), cfg, seed)
    Xtr, ytr = ps.X_train[tr], ps.y_train[tr]
    Ctr = ks.C_train[tr]
    Xte, yte = ps.X_test, ps.y_test

    model, history = train_cadvae(Xtr, Ctr, model_cfg, seed, device=device,
                                  n_constructs=n_constructs)

    Etr = encode(model, Xtr, device=device).astype(np.float64)
    Ete = encode(model, Xte, device=device).astype(np.float64)

    record: dict = {
        "dataset": cfg.data.name, "seed": seed, "beta": beta, "lambda_align": lam,
        "aligned_dims": aligned,
        "loss_final": history[-1], "loss_history": history,
        "downstream": {"primary": _fit_eval_heads(Etr, ytr, Ete, yte, seed),
                       **_task_heads(Etr, Ete, tasks, tr, seed)},
        "interpretability": interpretability_summary(Ete, ks.C_test, list(ks.names)),
    }
    pred_te = align_predict(model, Xte, device=device)
    if pred_te is not None:
        r2 = r2_score(ks.C_test, pred_te, multioutput="raw_values")
        record["alignment"] = {
            "r2_per_construct": {n: float(v) for n, v in zip(ks.names, r2, strict=True)},
            "r2_mean": float(np.mean(r2)), "r2_rfm_mean": float(np.mean(r2[:3])),
        }
    record["wall_time_s"] = round(time.time() - t0, 1)

    state = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
             "in_dim": Xtr.shape[1], "latent_dim": int(model_cfg.latent_dim),
             "hidden_dims": list(model_cfg.hidden_dims),
             "aligned_dims": aligned, "n_constructs": n_constructs,
             "seed": seed, "beta": beta, "lambda_align": lam}
    return record, state


def run_sweep(cfg: DictConfig) -> Path:
    sweep_dir = Path(cfg.results_dir) / "phase5" / f"{cfg.data.name}_sweep"
    RunLogger(sweep_dir, cfg)                   # provenance: config hash, commit, dirty
    runs_dir = sweep_dir / "runs"
    models_dir = sweep_dir / "models"
    runs_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    betas = [float(b) for b in cfg.sweep.betas]
    lambdas = [float(v) for v in cfg.sweep.lambdas]
    n_total = len(cfg.eval.seeds) * len(betas) * len(lambdas)
    n_done = 0
    for seed in cfg.eval.seeds:
        seed = int(seed)
        prepared = None                          # lazy: skip prepare() if seed complete
        for beta in betas:
            for lam in lambdas:
                tag = _run_tag(seed, beta, lam)
                out_json = runs_dir / f"{tag}.json"
                if out_json.exists():
                    n_done += 1
                    print(f"[{n_done}/{n_total}] {tag} SKIP (exists)", flush=True)
                    continue
                if prepared is None:
                    ps, ks = _prepare(cfg, seed)
                    prepared = (ps, ks, _tasks(ps, cfg))
                ps, ks, tasks = prepared
                record, state = run_point(ps, ks, tasks, cfg, seed, beta, lam,
                                          device=cfg.device)
                torch.save(state, models_dir / f"{tag}.pt")
                out_json.write_text(json.dumps(record, indent=1))
                n_done += 1
                mig = record["interpretability"]["mig"]
                pr = record["downstream"]["primary"]["gbt"]["pr_auc"]
                print(f"[{n_done}/{n_total}] {tag} done in {record['wall_time_s']}s "
                      f"(gbt pr_auc={pr:.4f}, mig={mig:.4f})", flush=True)

    manifest = {"grid": {"betas": betas, "lambdas": lambdas,
                         "seeds": [int(s) for s in cfg.eval.seeds]},
                "n_runs": n_total,
                "runs": sorted(p.name for p in runs_dir.glob("*.json"))}
    (sweep_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"SWEEP COMPLETE: {n_total} runs -> {sweep_dir}", flush=True)
    return sweep_dir


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_sweep(cfg)


if __name__ == "__main__":
    main()
