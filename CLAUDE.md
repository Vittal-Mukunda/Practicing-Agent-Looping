# CLAUDE.md — Construct-Aligned Disentangled VAE for Interpretable Buyer Personas

**Kickoff:** open a session in this repo and say: *"Read CLAUDE.md and begin at Phase 0. Confirm dataset + schema with me, propose the repo and config structure, then STOP at the Phase 0 gate."*

---

## Mission

Build a reproducible, **publication-grade** research codebase implementing and rigorously evaluating a **Construct-Aligned Disentangled Variational Autoencoder (CA-DVAE)** for interpretable, business-aligned buyer personas. Output must survive adversarial peer review. **Rigor, reproducibility, and honest evaluation outrank speed, cleverness, and feature count. A correct negative result beats an impressive unverified one.**

## Core scientific idea (governs every design choice)

Buyer personas in practice are unmeasurable and unstable. This project makes "persona quality" measurable via three properties:
1. **Downstream predictive lift (primary claim):** a good persona representation predicts real behavior (campaign response, next purchase, churn) better than incumbents.
2. **Interpretability:** latent axes align with *named* marketing constructs (RFM, price sensitivity, category affinity).
3. **Stability:** personas persist across seeds and resamples.

The VAE architecture is **not** claimed as novel. The contribution is the measurement reformulation + the construct-alignment mechanism + rigorous multi-dataset, multi-task, multi-seed evaluation, culminating in an **interpretability–performance trade-off curve** this area does not currently plot.

## Non-negotiable principles

- **Reproducibility first.** Global seed control; deterministic where feasible (document unavoidable GPU nondeterminism). Every experiment fully defined by a committed config. Pin all deps in a lockfile.
- **Config-driven.** Hydra + OmegaConf. No hardcoded hyperparameters. Sweeps via multirun. Every run writes resolved config + config hash + git commit + seed + full metrics to `results/`.
- **No data leakage — the #1 reviewer attack.** Temporal splits for behavioral data (predict future strictly from past); stratified for tabular. Representation learned on **train only**. Construct targets (RFM etc.) fit on **train only**, applied to val/test without refitting. Explicit leakage-guard unit tests.
- **Frozen-representation protocol.** Learn/fit representation on train → **freeze** → train a simple downstream head on frozen features → evaluate on held-out test. Downstream labels never touch representation learning.
- **Statistical rigor.** Every headline number = **mean ± std over ≥5 seeds**, plus a **paired significance test** (Wilcoxon or paired t) vs the strongest baseline. Never report a single run as a result.
- **Never fabricate.** Run real code; report real numbers; surface failures. If a baseline wins or a result is weak, say so and log it. Never invent or "expect" a metric.
- **Ask before assuming** on anything that affects results (see Execution Protocol).
- **Document as you go.** `docs/DECISIONS.md` (every non-obvious choice + rationale) and `docs/NOTEBOOK.md` (every experiment: config hash, seed, command, result, observation). Comment non-trivial algorithms (DEC, MIG, SAP, FactorVAE TC) with citations for the eventual methods section.

## Hardware & environment (hard limits — do not exceed)

- **GPU:** RTX 4050 laptop, **6 GB VRAM.** Models and batch sizes must fit comfortably; reject anything risking OOM.
- **CPU:** Ryzen 9 (many cores — exploit in preprocessing). **RAM: 16 GB — the binding constraint.** Cannot load multi-million-row event files whole.
- **Therefore:** preprocess behavioral data with **Polars lazy mode** (`scan_csv → filter → group_by → collect`). **Cache aggregated user×feature matrices to Parquet once**; load the small cached file for every experiment. Never re-run the heavy aggregation.
- **Phase 0 verifies** `torch.cuda.is_available()`, logs device + VRAM, confirms PyTorch uses the 4050 (catch silent CPU fallback).
- **Local-first logging:** JSON/CSV metrics + configs + checkpoints + figures under `results/`. W&B optional, OFF by default.

## Tech stack

Python, PyTorch (models), Polars (preprocessing), scikit-learn (baselines/metrics), Hydra+OmegaConf (config/sweeps), pytest (tests), matplotlib (figures). Pin versions in a lockfile (`uv`/`poetry`/`pip-tools` — pick one).

## Datasets

- **A — Customer Personality Analysis** (~2,240 × 29): interpretability story. **Stratified** splits.
- **B — eCommerce behavior** (multi-category store; millions of events): scale + downstream-lift story. **Temporal** splits.

**Phase 0:** confirm dataset choice + current license, then **inspect the actual schema** (do not assume columns). Emit a data card per dataset before modeling.

## Architecture — CA-DVAE

- **Base:** VAE over tabular/behavioral feature vectors. Latent dim configurable (8–32). Dense encoder/decoder. Trivially fits 6 GB.
- **Component 1 — construct-alignment head:** auxiliary head trains a *designated subset* of latent dims to predict named constructs (RFM components, price-sensitivity index, category-affinity vector). Makes axes named/interpretable. (Delta over prior single-axis latent steering: multiple interpretable constructs, jointly.)
- **Component 2 — disentanglement penalty:** β-VAE KL weighting on the remaining free dims (optional FactorVAE total-correlation term if Phase 4 finishes early).
- **Loss:** `L = reconstruction + β·KL + λ·construct_alignment`. **β and λ are swept** to generate the trade-off curve. Two independently ablatable components.

## Baselines (implement ALL; beat on downstream lift or report honestly)

| Model | Purpose |
|---|---|
| RFM + K-means | Incumbent floor |
| **Autoencoder + K-means** | **The hard baseline you must beat** |
| Deep Embedded Clustering (DEC) | Hard neural baseline |
| GMM on AE embeddings | Hard baseline |
| Vanilla β-VAE | Ablation = proposed minus alignment |
| VAE + alignment, no disentanglement | Ablation = proposed minus disentanglement |
| **CA-DVAE (full)** | **Proposed** |

## Evaluation protocol (implement precisely)

- Frozen-representation protocol with **two** downstream heads: regularized logistic regression AND gradient-boosted trees.
- **Tasks:** campaign response (A); next-purchase-category / repeat purchase (B); churn/dormancy (engineered from B). **Metrics:** ROC-AUC, PR-AUC, top-k.
- **Clustering:** silhouette, Davies–Bouldin, Calinski–Harabasz.
- **Interpretability:** MIG, SAP, per-axis alignment score.
- **Stability:** ARI/NMI of persona assignments across ≥5 seeds; bootstrap-resample persistence.
- **Headline figure:** interpretability–performance trade-off curve (sweep β/λ; downstream AUC vs MIG/alignment; mark the sweet spot).
- **Persona cards:** latent-traversal visualizations (move one aligned axis, show the reconstructed customer change).

## Repository structure (propose at Phase 0; adjust to my feedback)

```
repo/
  CLAUDE.md  README.md  pyproject.toml/lockfile
  configs/                 # Hydra: model, data, experiment, sweep
  src/<pkg>/{data,models,eval,viz,utils}/
  data/{raw,interim,processed}/   # processed = Parquet caches (gitignored)
  experiments/  results/  tests/
  docs/DECISIONS.md  docs/NOTEBOOK.md
```

## Phased plan with STOP gates (never pass a gate without my sign-off)

- **Phase 0 — Setup.** Env + lockfile; CUDA/device check; repo scaffold; confirm dataset + license + real schema; Hydra skeleton; local logging; pytest harness. **GATE:** approve structure + data cards.
- **Phase 1 — Data.** Polars lazy preprocessing + Parquet cache for BOTH datasets; leakage-safe splits; data cards; pipeline + leakage tests pass. **GATE:** review split logic + data cards.
- **Phase 2 — Constructs.** Extract RFM / price-sensitivity / category-affinity leakage-safe (train-only fit). **GATE:** review construct definitions.
- **Phase 3 — Baselines.** Implement + run ALL baselines through the frozen-rep protocol; log numbers. **GATE:** the bar to beat is fixed *before* any CA-DVAE work.
- **Phase 4 — CA-DVAE + ablations.** Model, loss terms, alignment head; smoke tests; one sanity run with logged curves. **GATE:** review model + sanity run.
- **Phase 5 — Campaign.** Full multi-seed × β/λ sweep via Hydra multirun (overnight batches); all metrics logged with hashes. **GATE:** review raw results.
- **Phase 6 — Analysis + figures.** Trade-off curve, stability, ablations, persona cards, significance tests, publication figures. **GATE:** review figures + tables.
- **Phase 7 — Reproduction package.** README with exact commands/seeds/configs; verify a fresh clone reproduces key numbers; finalize DECISIONS + NOTEBOOK.

## Out of scope (do NOT build — future-work notes only)

VAE+GNN (VRAM risk, occupied territory); raw-sequence/Transformer/RNN encoders over event streams; any paid/cloud compute; any architecture risking >6 GB VRAM; any scope expansion before Phases 0–6 are complete and written up.

---

## EXECUTION PROTOCOL (mandatory)

Execute as a deterministic phased research agent, not a one-shot coding assistant.

### 1. Phase isolation
One phase at a time. Never start the next phase until the current one is fully complete and I have signed off at the gate. Do not skip, merge, or partially execute phases. If a dependency for the current phase is missing, stop and ask.

### 2. Core loop (per task/subtask)

```
PLAN → implement smallest logical unit → run relevant tests/commands →
inspect outputs & logs → find failures/warnings/TODOs/edge cases →
fix → re-verify → repeat until exit criteria met → mark complete → next
```

Never assume something works because it compiles or runs without error. Verify every claim with evidence (a printed metric, a passing test, an inspected output).

### 3. Bounded retry + integrity guard (research-specific — critical)
Each verify-fix loop has a **retry budget of 3–5 attempts.** If a check still fails after the budget, **STOP and surface it to me** with the failing output and your diagnosis. **Never** make a check pass by weakening, deleting, skipping, or loosening a test, or by altering a metric, threshold, split, or result. **Making a check pass is not the goal; correct evidence is.** A correct *failing* check (e.g., "the model does not beat this baseline") is a legitimate, reportable stopping point — not a bug to grind on.

### 4. Verification cadence (right-sized for ML)
**Cheap checks — run after every significant change:** unit tests, lint, type-check, config-load, and leakage-*guard* unit tests.
**Expensive checks — run at phase gates and via dedicated commands, not every edit:** full reproducibility re-runs (fixed seed, compare numbers), the full sweep, and end-to-end leakage audits.
Concrete definitions for this repo: **type-check** = pyright/mypy in basic mode on `src/` (do not add heavy stubs to NumPy-heavy code); **integration test** = one full frozen-representation run on a tiny sampled subset, asserting the pipeline completes and emits all expected metrics.
If any check fails: diagnose → fix → re-run → repeat (within the retry budget).

### 5. Skeptical numbers pass
A green test suite is necessary but not sufficient. Tests only check what we thought to check; the failure that sinks this paper is a *silently wrong metric* (leakage the tests missed, AUC on the wrong split) that passes everything and looks publishable. Periodically — and always before a gate — ask "do these numbers actually make sense?" (sane ranges, baseline ordering plausible, no too-good-to-be-true jumps) and flag anything suspicious.

### 6. Autonomy scope
Within a phase, work autonomously between small **engineering** decisions — don't ask permission for routine implementation choices. **But any decision that affects the scientific results is consequential: pause and ask, even if it feels small.** Consequential includes feature definitions, construct bucketing, split boundaries/ratios, label definitions, and metric choices. When unsure whether a decision is consequential, treat it as consequential.

### 7. Stop conditions
Stop and ask me only when: (1) the phase is fully complete (→ gate); (2) a consequential/scientific decision needs my input; (3) a missing dependency blocks progress; (4) requirements contradict each other; or (5) **results suggest the scientific approach itself needs rethinking** (e.g., the proposed model cannot beat a baseline) — surface it, do not grind.

### 8. No forward progress with known issues
Every issue found in a phase must be fixed, explicitly approved by me, or logged as a documented future-work item — never silently postponed. If it can't be resolved within the retry budget, surface it (per §3).

### 9. Progress tracking + session hygiene
Maintain a live checklist (Pending / In Progress / Completed / Blocked) per phase, updated after every completed task. **Because each session starts fresh, update the checklist and `docs/NOTEBOOK.md` BEFORE ending any session**, so a cold start can reconstruct exactly where things stand. Commit at every gate.

### Default behavior

```
Plan → Implement → Verify → Fix (≤ budget) → Verify → Document →
Verify entire phase → STOP at gate → Wait for approval
```

Never advance merely because code has been written.

---

## Definition of done

A fresh clone, following the README, reproduces: all baseline + CA-DVAE metrics (mean ± std over seeds) on both datasets, the trade-off curve, the stability analysis, and every paper figure — configs, seeds, and commit hashes committed, all tests passing, DECISIONS + NOTEBOOK complete. Guiding standard: **could a skeptical reviewer re-run this and get the same numbers, and is every claim falsifiable and backed by logged evidence?**

## Start here

Do NOT write modeling code yet. First: (1) confirm the eCommerce dataset + license; (2) inspect actual schemas and emit data cards; (3) propose the repo structure and Hydra config schema for approval. Then STOP at the Phase 0 gate.
