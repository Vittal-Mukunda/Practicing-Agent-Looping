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
| 1.1 | Download `2019-Nov.csv` (9 GB) + verify | **Completed** — 2019-Nov.csv on disk; combined 109,950,743 rows verified |
| 1.2 | Consequential-decision block (D-013..017): A split/cleaning; B windows/universe/labels/features | **Completed** — all recorded in DECISIONS.md, config-driven, gate-reviewable |
| 1.3 | Dataset A preprocessing (clean, stratified split, Parquet cache, feature matrix) | **Completed** — `src/cadvae/data/personality.py`; 2240×34; verified (stratification, no-NaN, determinism) |
| 1.4 | Dataset B preprocessing (streaming user×feature aggregation Oct+Nov, temporal split, Parquet cache) | **Completed** — `src/cadvae/data/ecommerce.py`; 2.55M×27; ~200 s streaming build cached |
| 1.5 | Leakage-guard unit tests (train-only fit; temporal ordering; label-window separation; user-disjoint) | **Completed** — `tests/test_leakage.py`, 8 tests incl. perturbation proofs + real-cache cross-check |
| 1.6 | Pipeline/integration test (contract + emitted artifacts) | **Completed** — `tests/test_pipeline.py`; full suite **25 passed**, ruff+mypy clean |
| 1.7 | Update both data cards (post-processing shapes, split sizes, Nov facts, corrected BF rationale) | **Completed** |
| 1.8 | GATE: owner reviews split logic + leakage tests + data cards | **PASSED 2026-07-10** (owner sign-off, session 7) |
## Phase 2 — Constructs  → GATE: owner reviews construct definitions

| # | Task | Status |
|---|------|--------|
| 2.1 | Construct definitions (RFM / price-sensitivity / category-affinity) per dataset | **Completed** — D-018; encoded in `constructs:` config blocks |
| 2.2 | Extraction module, leakage-safe (computed pre-standardization; same per-seed split as prepare; standardized train-only) | **Completed** — `src/cadvae/data/constructs.py` |
| 2.3 | Verify on real data (shapes, alignment, sane raw values) | **Completed** — A: 10 targets, affinity sums to 1, deal-reliance∈[0,1]; B: 17 targets, affinity sum 0.69 ≡ known-category rate |
| 2.4 | Leakage-guard + definition tests | **Completed** — `tests/test_constructs.py` (4); full suite **29 passed**, ruff+mypy clean |
| 2.5 | GATE: owner reviews construct definitions | **PASSED 2026-07-10** (owner sign-off, session 7) |
## Phase 3 — Baselines  → GATE: bar to beat fixed before any CA-DVAE work

| # | Task | Status |
|---|------|--------|
| 3.1 | Protocol + K/latent/subsample decisions (D-019/020) | **Completed** — config-driven |
| 3.2 | Models: AE (`models/ae.py`), DEC (`models/dec.py`) | **Completed** |
| 3.3 | Baselines + frozen-rep harness + metrics (`eval/protocol.py`, `eval/metrics.py`) — RFM, RFM+KMeans, AE, AE+KMeans, GMM-on-AE, DEC × {logreg, HistGBT} | **Completed** |
| 3.4 | Multi-seed aggregation + significance (`eval/stats.py`), Hydra runner (`eval/run_baselines.py`) | **Completed** — `significance_vs_best` reports **both** paired Wilcoxon AND paired t (Wilcoxon two-sided p-floor = 2^-(n-1) = 0.0625 at n=5, so the t-test carries significance at 5 seeds; ≥6 seeds recommended for Phase 5 headline claims — D-019 updated) |
| 3.5 | Smoke test (`tests/test_eval_smoke.py`) | **Completed** — full suite **31 passed**, ruff+mypy clean |
| 3.6 | Run Dataset A (5 seeds) — bar to beat | **Completed** — AE embedding PR-AUC 0.532±0.006 (gbt) / 0.572±0.042 (logreg); AE beats every other method (paired-t p<0.003) |
| 3.7 | Run Dataset B (5 seeds, subsample 200k) — bar to beat | **Completed 2026-07-09** — RFM gbt PR-AUC **0.1947±0.0044** (best); AE gbt 0.1724±0.0056; RFM > AE (paired-t p=3.6e-4). Honest finding: RFM ≥ AE on behavioral data |
| 3.8 | GATE: owner reviews baseline numbers (bar fixed) | **PASSED 2026-07-10** (owner sign-off, session 7) — audited-protocol bars: A **pca 0.5677**, B **rfm 0.1953** (gbt, 6 seeds); **multi-task framing of the lift claim blessed** (primary + churned + next_category) |
## Phase 4 — CA-DVAE + ablations  → GATE: owner reviews model + sanity run

| # | Task | Status |
|---|------|--------|
| 4.1 | Model: CA-DVAE (`models/cadvae.py`) — VAE + linear construct-alignment head + free/aligned latent split | **Completed** — D-021 |
| 4.2 | Loss terms `recon + β·KL_free + w_a·KL_aligned + λ·align`; ablations as pure config points (β-VAE=λ0, VAE+align=β1, full) | **Completed** — D-022 |
| 4.3 | Config `model/cadvae.yaml` finalized (aligned_dims auto-rule, min_free_dims, beta/lambda/aligned_kl_weight) | **Completed** |
| 4.4 | Smoke tests (`tests/test_cadvae_smoke.py`, 6) — train/shapes/determinism/ablations/loss split | **Completed** — full suite **37 passed**, ruff+mypy clean |
| 4.5 | Sanity runner (`eval/run_cadvae_sanity.py`) — logged curves + frozen-rep heads + alignment R² | **Completed** |
| 4.6 | Sanity run A (GPU, 200 ep, seed 7) | **Completed** — gbt PR-AUC 0.479 vs AE bar 0.532 (Δ−0.053, expected); align RFM R²=0.90; exit 0 |
| 4.7 | Sanity run B (GPU, 50k/40 ep, seed 7) — pipeline + timing | **Completed** — gbt PR-AUC 0.150 vs RFM bar 0.195 (Δ−0.037); align RFM R²=0.68; ~110 s → 200k run ≈3–4 min |
| 4.8 | GATE: owner reviews model design + sanity run | **PASSED 2026-07-10** (owner sign-off, session 7) |
## Full repo audit (2026-07-09/10, pre-gate-review) — session 7

| # | Task | Status |
|---|------|--------|
| A.1 | Protocol upgrades: 6 seeds, multi-task (B: churned + next_category), raw ceiling + PCA, method subsetting, tie-aware prec@k (D-023/024/025) | **Completed** — verified after session interruption |
| A.2 | Bit-reproducibility: threadpool_limits on rep-feeding fits, deterministic DEC distance/permutation, sorted row order (D-028) | **Completed** |
| A.3 | Interpretability metrics MIG/SAP/axis-alignment + ground-truth tests (D-026) | **Completed** — `eval/interpretability.py`, 6 tests |
| A.4 | Fix streaming-engine crash in next-category label agg (filter-inside-sort_by) | **Completed 2026-07-10** — single-sorted-column form; leakage tests 8/8 |
| A.5 | Remove silent CPU fallback in AE/CA-DVAE trainers (D-027) | **Completed 2026-07-10** — `resolve_device` raises |
| A.6 | Audit remaining modules (cadvae, ae, constructs, sanity runner, utils, configs) | **Completed 2026-07-10** — no further defects; stability metrics correctly absent (Phase 5/6 work) |
| A.7 | Documentation debt: DECISIONS D-023..028, NOTEBOOK session 7, this section | **Completed 2026-07-10** |
| A.8 | Full cheap-check suite | **Completed 2026-07-10** — pytest 45 passed, ruff clean, mypy clean (21 files) |
| A.9 | Re-run Phase 3 baselines under the audited protocol (bars were stale) | **Completed 2026-07-10** — new bars: A **pca 0.5677**, B **rfm 0.1953** (6 seeds, multi-task); seeds 0–4 integrity-checked vs old runs; old evidence preserved in `*_5seed_preaudit` |
| A.10 | Fix latent multiclass-head crash (HistGBT auto early-stopping stratified split vs singleton classes) | **Completed 2026-07-10** — `early_stopping=False` on the multiclass head only + regression test; suite **46 passed**, ruff+mypy clean |

## Phase 5 — Campaign  → GATE: owner reviews raw sweep results

| # | Task | Status |
|---|------|--------|
| 5.1 | Sweep grid + runner design (D-029): 6β×4λ×6 seeds, resume-skip, saved state_dicts | **Completed 2026-07-10** — `eval/run_cadvae_sweep.py`; suite 49 passed |
| 5.2 | Dry runs (A + B real caches) + protocol-scale timing anchor (B run = 179.5 s) | **Completed 2026-07-10** — resume verified; B sweep ≈7.5 h, A ≈2 h |
| 5.3 | Overnight sweep, Dataset B (144 runs) | **Pending — owner launches** (commands in NOTEBOOK) |
| 5.4 | Overnight sweep, Dataset A (144 runs) | **Pending — owner launches** |
| 5.5 | GATE: owner reviews raw sweep results | Pending 5.3/5.4 |
## Phase 6 — Analysis + figures  → GATE: owner reviews figures + tables

| # | Task | Status |
|---|------|--------|
| 6.1 | Stability metrics (ARI/NMI cross-seed, bootstrap persistence) — D-030 | **Code completed 2026-07-10** — `eval/stability.py`; ground-truth tests |
| 6.2 | Sweep analysis (aggregate, Pareto, sweet-spot rule, paired tests) — D-031 | **Code completed 2026-07-10** — `eval/sweep_analysis.py` |
| 6.3 | Figures: trade-off curve, stability bars, persona cards, tables — D-032 | **Code completed 2026-07-10** — `viz/figures.py` |
| 6.4 | Orchestrator (`eval/run_phase6.py`) — analysis.json + figures + tables | **Code completed 2026-07-10** — e2e-tested on faked sweep artifacts |
| 6.5 | Run Phase 6 on real sweep output (A + B) | **Blocked on 5.3/5.4** — first real run = integration test; D-030..032 may be tuned then (owner pre-authorized) |
| 6.6 | GATE: owner reviews figures + tables | Pending 6.5 |

## Phase 7 — Reproduction package  → verify fresh-clone reproduction

| # | Task | Status |
|---|------|--------|
| 7.1 | `eval/repro_check.py` — re-run pinned seed, diff vs recorded records (exit 1 on mismatch) | **Code completed 2026-07-10** — comparator unit-tested |
| 7.2 | README rewritten as reproduction guide (all commands, evidence table) | **Completed 2026-07-10** |
| 7.3 | Run repro_check on both datasets; fresh-clone walkthrough | **Blocked on Phase 5/6 runs** |
| 7.4 | Hygiene: add `threadpoolctl` as direct dep at next `uv lock` (network) | Pending |
| 7.5 | Finalize DECISIONS + NOTEBOOK; final commit | Pending 7.3 |

