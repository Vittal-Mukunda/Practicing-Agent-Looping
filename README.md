# CA-DVAE — Construct-Aligned Disentangled VAE for Interpretable Buyer Personas

Reproducible research codebase evaluating whether a construct-aligned, disentangled
VAE representation of customers (a) predicts real behavior better than incumbent
persona representations, (b) yields latent axes aligned with named marketing
constructs (RFM, price sensitivity, category affinity), and (c) is stable across
seeds and resamples. See [CLAUDE.md](CLAUDE.md) for the full research protocol.

**Status: Phase 0 (setup) — awaiting gate sign-off.** No modeling code exists yet, by design.

## Setup

Requires: Windows, an NVIDIA GPU (developed against an RTX 4050 laptop, 6 GB VRAM), Python 3.12.

```powershell
python -m pip install uv          # or: winget install astral-sh.uv
python -m uv sync                 # creates .venv from uv.lock (exact pinned deps, CUDA 12.4 torch)
python -m uv run python -m cadvae.utils.device   # GPU verification — must print cuda_available: true
python -m uv run pytest           # environment + config + logging tests
```

## Data

Datasets are **not** committed. See [data/README.md](data/README.md) for download
instructions and the expected layout under `data/raw/`.

## Repository layout

```
configs/          Hydra configs (data, model; experiment/sweep groups arrive Phase 3+)
src/cadvae/       Python package: data / models / eval / viz / utils
data/             raw + interim + processed (gitignored; processed = Parquet caches)
experiments/      run scripts and sweep launchers (Phase 3+)
results/          per-run resolved config, config hash, git commit, metrics (gitignored)
tests/            pytest: environment, config, seeding, run logging (leakage guards arrive Phase 1)
docs/             DECISIONS.md, NOTEBOOK.md, CHECKLIST.md, data cards
```

## Reproduction

Exact reproduction commands, seeds, and config hashes for every reported number
will be added here as phases complete (see CLAUDE.md, Definition of Done).
