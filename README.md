# CA-DVAE — Construct-Aligned Disentangled VAE for Interpretable Buyer Personas

Reproducible research codebase evaluating whether a construct-aligned, disentangled
VAE representation of customers (a) predicts real behavior better than incumbent
persona representations, (b) yields latent axes aligned with named marketing
constructs (RFM, price sensitivity, category affinity), and (c) is stable across
seeds and resamples. See [CLAUDE.md](CLAUDE.md) for the full research protocol and
[docs/DECISIONS.md](docs/DECISIONS.md) for every consequential choice (D-001…).

**Status: all phase code implemented (Phases 0–7). Phases 5–6 runs pending**
(see [docs/CHECKLIST.md](docs/CHECKLIST.md) for live state). Recorded evidence so
far: Phase-3 baseline bars on both datasets (6 seeds, multi-task), Phase-4 sanity
runs. Analysis/figure code may be revised after the sweep lands (per protocol,
any revision is a new DECISIONS entry).

## Setup

Requires: Windows, an NVIDIA GPU (developed against an RTX 4050 laptop, 6 GB VRAM), Python 3.12.

```powershell
python -m pip install uv          # or: winget install astral-sh.uv
python -m uv sync                 # creates .venv from uv.lock (exact pinned deps, CUDA 12.4 torch)
.venv\Scripts\python.exe -m cadvae.utils.device   # GPU check — must print cuda_available: true
.venv\Scripts\python.exe -m pytest                # full test suite (leakage guards included)
```

(If `uv run` fails with a trampoline error on Windows, invoke the venv interpreter
directly as above — same environment.)

## Data

Datasets are **not** committed. See [data/README.md](data/README.md) for download
instructions and the expected layout under `data/raw/`. Then build the caches:

```powershell
# Dataset A cache is built on demand by prepare(); Dataset B needs one streaming pass (~4 min):
.venv\Scripts\python.exe -c "from omegaconf import OmegaConf; from cadvae.data.ecommerce import cache_user_table; cache_user_table(OmegaConf.load('configs/data/ecommerce.yaml'))"
```

## Pipeline — exact commands per phase

All runs are config-driven (Hydra); every run directory records the resolved
config, its hash, the git commit, dirty flag, and seed. Seeds: `eval.seeds =
[0..5]` (D-023). Dataset B always takes the D-020 protocol overrides shown below.

```powershell
# Phase 3 — baselines (fixes the bar; multi-task records)
.venv\Scripts\python.exe -m cadvae.eval.run_baselines data=personality
.venv\Scripts\python.exe -m cadvae.eval.run_baselines data=ecommerce eval.train_subsample=200000 model.max_epochs=40

# Phase 4 — single-point sanity run (gate evidence, not a result)
.venv\Scripts\python.exe -m cadvae.eval.run_cadvae_sanity data=personality

# Phase 5 — the beta x lambda sweep (resumable; ~2 h for A, ~7.5 h for B on the 4050)
powercfg /change standby-timeout-ac 0     # once, before an overnight run
.venv\Scripts\python.exe -m cadvae.eval.run_cadvae_sweep data=personality
.venv\Scripts\python.exe -m cadvae.eval.run_cadvae_sweep data=ecommerce eval.train_subsample=200000 model.max_epochs=40

# Phase 6 — analysis, trade-off curve, stability, persona cards (no training)
.venv\Scripts\python.exe -m cadvae.eval.run_phase6 data=personality
.venv\Scripts\python.exe -m cadvae.eval.run_phase6 data=ecommerce eval.train_subsample=200000

# Phase 7 — reproduction check (re-runs one pinned seed, diffs vs recorded records)
.venv\Scripts\python.exe -m cadvae.eval.repro_check data=personality
.venv\Scripts\python.exe -m cadvae.eval.repro_check data=ecommerce eval.train_subsample=200000 model.max_epochs=40
```

## Recorded headline evidence (updated at each gate)

| Item | Value | Where |
|---|---|---|
| Bar, Dataset A (primary PR-AUC, gbt, 6 seeds) | **pca 0.5677 ± 0.0729** | `results/phase3/personality_baselines/` |
| Bar, Dataset B (primary PR-AUC, gbt, 6 seeds) | **rfm 0.1953 ± 0.0035** | `results/phase3/ecommerce_baselines/` |
| B multi-task: next-category acc@1 | ae_kmeans 0.5550 ≫ rfm 0.4070 (p=0.017) | same |
| B multi-task: churn PR-AUC | ae 0.8620 > rfm 0.8498 (p=1.7e-8) | same |
| Sweep results / trade-off curve / stability | pending Phase 5–6 runs | `results/phase5/`, `results/phase6/` |

## Repository layout

```
configs/          Hydra configs: root (eval/sweep/phase6/repro blocks), data/, model/
src/cadvae/
  data/           Polars preprocessing, leakage-safe splits, constructs (Phase 1–2)
  models/         ae.py, dec.py, cadvae.py (Phase 3–4)
  eval/           protocol, metrics, stats, interpretability, sweep + analysis runners
  viz/            figures.py — trade-off curve, stability bars, persona cards
  utils/          seeding (determinism), device (no silent CPU fallback), run logging
data/             raw + interim + processed (gitignored; processed = Parquet caches)
results/          per-run evidence (gitignored; numbers are recorded in docs/ + README)
tests/            pytest — leakage guards, protocol smoke, metric ground-truth, phase-6 e2e
docs/             DECISIONS.md, NOTEBOOK.md, CHECKLIST.md, data cards
```

## Definition of done

A fresh clone, following this README, reproduces all reported numbers (mean ± std
over 6 seeds), the trade-off curve, the stability analysis, and every figure —
verified by `repro_check` (exit 0). The pipeline is bit-reproducible on a single
machine (D-028); use `repro.atol` for cross-machine tolerance.
