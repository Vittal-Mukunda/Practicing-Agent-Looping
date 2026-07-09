# Live checklist (per CLAUDE.md §9)

Statuses: Pending / In Progress / Completed / Blocked.
A cold-start session reconstructs project state from this file + NOTEBOOK.md.

## Phase 0 — Setup  → GATE: owner approves structure + data cards

| # | Task | Status |
|---|------|--------|
| 0.1 | git init, baseline CLAUDE.md commit (`64415ec`) | Completed |
| 0.2 | uv installed; `pyproject.toml` + `uv.lock` (48 pkgs, win32-only, torch 2.6.0+cu124) | Completed |
| 0.3 | `.venv` synced (all groups incl. dev) | Completed (uv sync exit 0, torch 2.6.0+cu124 installed) |
| 0.4 | CUDA verification: `python -m cadvae.utils.device` shows RTX 4050 + VRAM, real GPU matmul | Completed 2026-07-09 (RTX 4050 Laptop GPU, 6.0 GB VRAM, cuda 12.4, cudnn 90100, gpu_matmul_ok=true) |
| 0.5 | Repo scaffold: configs/ src/cadvae/{data,models,eval,viz,utils} tests/ docs/ data/ experiments/ | Completed |
| 0.6 | Hydra config skeleton (root + data×2 + model×2), null-means-undecided convention | Completed (tests pending 0.3) |
| 0.7 | Local run logging (`RunLogger`: resolved config, config hash, git commit, seed, metrics.jsonl) | Completed (tests pending 0.3) |
| 0.8 | pytest harness green (environment, config, seeding, run-logging) | Completed 2026-07-09 (15 passed) |
| 0.9 | Lint (ruff) + type-check (mypy basic) clean on src/ | Completed 2026-07-09 (ruff clean after UP017 autofix in run_logging.py; mypy: no issues in 9 files; pytest re-run green after fix) |
| 0.10 | Datasets downloaded to `data/raw/` | **Blocked — no Kaggle credentials on this machine; owner action (see data/README.md)** |
| 0.11 | Schema inspection (actual files) + data cards emitted | **Blocked — depends on 0.10; skeletons at docs/data_cards/** |
| 0.12 | Dataset licenses confirmed on Kaggle pages | Pending — owner action at gate |
| 0.13 | GATE sign-off: structure + data cards + open decisions (D-005, D-008, D-009, D-011) | Pending |

## Phase 1 — Data — not started (gated)
## Phase 2 — Constructs — not started (gated)
## Phase 3 — Baselines — not started (gated)
## Phase 4 — CA-DVAE + ablations — not started (gated)
## Phase 5 — Campaign — not started (gated)
## Phase 6 — Analysis + figures — not started (gated)
## Phase 7 — Reproduction package — not started (gated)
