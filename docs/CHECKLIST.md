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
| 0.10 | Datasets downloaded to `data/raw/` | Completed for gate 2026-07-09 — A (`marketing_campaign.csv`, CC0, sha256 recorded) + B `2019-Oct.csv` (42.4M rows, 5.67 GB) on disk. **B `2019-Nov.csv` deferred to Phase 1 start** (identical schema; no gate value in pulling 9 GB now) |
| 0.11 | Schema inspection (actual files) + data cards emitted | **Completed 2026-07-09** — both cards filled from real inspection (A: full polars profile; B-Oct: streaming lazy scan). Key finds: A imbalance 14.9%, constant Z-cols, Income nulls/outlier; B purchase-rate 1.75%, category_code 31.84% null, **no remove_from_cart in Oct** |
| 0.12 | Dataset licenses confirmed on Kaggle pages | **Completed** — A = CC0-1.0 (first-hand); B = copyright-authors → **resolved: proceed under attribution + non-redistribution posture** (D-011) |
| 0.13 | GATE sign-off: structure + data cards + open decisions | **PASSED 2026-07-09 (owner-delegated, D-012)** — D-004/005/008 locked, D-009 resolved, D-011-B resolved |

## Phase 1 — Data  → GATE: owner reviews split logic + data cards
Owner delegated forward progress (D-012); consequential choices made + documented
as D-entries, config-driven and revertable. Owner reviews at the Phase 1 gate.

| # | Task | Status |
|---|------|--------|
| 1.1 | Download `2019-Nov.csv` (Dataset B month 2, 9 GB) + verify | Pending |
| 1.2 | Consequential-decision block: A split ratios + stratify col; B temporal boundaries; label definitions (campaign / repeat / next-cat / churn); feature definitions | Pending → D-entries |
| 1.3 | Dataset A preprocessing: clean (Income/Year_Birth/Marital/Z-cols/Dt_Customer), stratified split, Parquet cache, feature matrix | Pending |
| 1.4 | Dataset B preprocessing: Polars lazy `scan→filter→group_by→collect(streaming)` user×feature aggregation (Oct+Nov), temporal split, Parquet cache | Pending |
| 1.5 | Leakage-guard unit tests (train-only fit; no test rows in train; temporal ordering; construct targets not refit on val/test) | Pending |
| 1.6 | Pipeline/integration test: tiny sampled subset runs end-to-end, emits expected artifacts | Pending |
| 1.7 | Update both data cards (post-processing shapes, split sizes, Nov facts, user-overlap) | Pending |
| 1.8 | GATE: owner reviews split logic + leakage tests + data cards | Pending |
## Phase 2 — Constructs — not started (gated)
## Phase 3 — Baselines — not started (gated)
## Phase 4 — CA-DVAE + ablations — not started (gated)
## Phase 5 — Campaign — not started (gated)
## Phase 6 — Analysis + figures — not started (gated)
## Phase 7 — Reproduction package — not started (gated)
