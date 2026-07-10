# NOTEBOOK.md — experiment log

Every experiment records: config hash, seed, command, result, observation.
No experiments have been run yet (Phase 0 = setup only, by design).

---

## 2026-07-09 — Session 1 (Phase 0)

**Done:**
- Repo bootstrapped from empty directory. Baseline commit `64415ec` (CLAUDE.md only).
- Environment: Python 3.12.10, uv 0.11.28, `uv.lock` resolved (48 packages,
  win32-only), torch 2.6.0+cu124 from the PyTorch CUDA index.
- Scaffold: Hydra config skeleton, `cadvae` package (utils implemented:
  seeding / device / run_logging; data/models/eval/viz are placeholders by design),
  pytest harness (4 test modules), docs (DECISIONS, CHECKLIST, data-card skeletons).
- `uv sync --all-groups` completed (exit 0): .venv created, torch 2.6.0+cu124
  installed. Lockfile verified to pin the CUDA wheel (cp312, win_amd64, sha256),
  not a CPU wheel.

**Session paused by owner BEFORE verification ran. Nothing is verified yet:
no CUDA check, no pytest, no lint/type-check, scaffold not committed. Do not
mark 0.4/0.8/0.9 complete without running them. First actions on resume:**
```powershell
python -m uv run python -m cadvae.utils.device   # must show RTX 4050, cuda_available true
python -m uv run pytest                          # all 4 test modules must pass
python -m uv run ruff check src tests
python -m uv run mypy
# then: git add -A; git commit (Phase 0 scaffold); paste evidence here; gate summary
```

**Blocked / pending (owner):**
- Datasets not on disk; no Kaggle credentials on machine → schema inspection and
  final data cards blocked (checklist 0.10/0.11). Download instructions: data/README.md.
- License confirmation on both Kaggle pages (D-011).
- Open decisions at gate: D-005 (results/ gitignored), D-008 (GBT head choice),
  D-009 (OneDrive location — recommend relocating before the 15 GB download).

**Observations:**
- `nvidia-smi` requires admin on this machine (fails with a permissions error even
  unsandboxed) — GPU facts are gathered via torch/CUDA runtime instead, which is
  what matters for training anyway.
- pip's first attempt to install uv hit a network read timeout; succeeded on retry
  with `--default-timeout=120`. Worth remembering for the big dataset downloads.

---

## 2026-07-09 — Session 2 (Phase 0 verification)

**Verification evidence (all four resume commands run):**
- `python -m uv run python -m cadvae.utils.device` → torch 2.6.0+cu124,
  cuda_available=true, NVIDIA GeForce RTX 4050 Laptop GPU, 6.0 GB VRAM,
  cuda runtime 12.4, cudnn 90100, compute capability 8.9, gpu_matmul_ok=true
  (9.1 MB allocated after matmul). No silent CPU fallback.
- `pytest` → **15 passed** (7.03 s).
- `ruff check src tests` → 1 finding (UP017, `timezone.utc` → `datetime.UTC` in
  `run_logging.py`); `--fix` applied (3 mechanical fixes incl. import cleanup),
  re-check clean. pytest re-run after fix: 15 passed.
- `mypy` → Success: no issues found in 9 source files.

**Result:** checklist 0.4 / 0.8 / 0.9 flipped to Completed. Scaffold committed
(this commit). Phase 0 remaining items are all owner-blocked: 0.10 datasets
(Kaggle creds), 0.11 data cards (depends on 0.10), 0.12 licenses (D-011),
0.13 gate sign-off (D-005, D-008, D-009 OneDrive, D-004 git identity).

---

## 2026-07-09 — Session 3 (Phase 0 data acquisition + schema inspection)

Owner provided a Kaggle API token (`KGAT_…`) → unblocked 0.10/0.11.

**Auth + downloads (all commands + gotchas in data/README.md):**
- Token → `KAGGLE_API_TOKEN` env var, persisted to `~/.kaggle/access_token`
  (outside repo, chmod 600). kaggle run via `uv tool run --from kaggle` (isolated;
  NOT a project dep — `uv run kaggle` tried to rebuild the editable pkg and hit a
  Windows "Access is denied" lock on `.venv/.../cadvae-0.1.0.dist-info`). For all
  Polars inspection I called `.venv/Scripts/python.exe` directly to avoid that lock.
- **Dataset A**: `marketing_campaign.csv`, 220,188 B, sha256
  `cd0affa36b1b981e80ba0e27767e9b3ab723f7a3dba948f722af81abc6b990ea`.
  License printed by CLI: **CC0-1.0** (resolves A-half of D-011).
- **Dataset B**: `2019-Oct.csv` only (5,668,612,855 B). `-f` single-file download
  ignores `--unzip`; extracted with `unzip`. License printed: **`copyright-authors`**
  (restrictive — contradicts HF-mirror wording; D-011 now needs owner review).
  Nov deferred to Phase 1 (same schema; 9 GB, no gate value now).

**Schema inspection (real, not assumed — scripts in session scratchpad):**
- A (`read_csv`, tab-sep): 2240×29 confirmed. `Response` pos-rate 0.149. `Income`
  24 nulls + 666,666 outlier (p99 94,472). `Year_Birth` 3 rows <1920. `Marital_Status`
  junk levels Absurd/Alone/YOLO (7 rows). `Z_CostContact`/`Z_Revenue` constant.
- B-Oct (`scan_csv` + `engine="streaming"`, 5.67 GB in ~59 s, bounded RAM): 9 cols,
  **42,448,764 rows**. Nulls: category_code 31.84%, brand 14.40%, user_session 2.
  event_type = view 96.07% / cart 2.18% / **purchase 1.75%** (→ PR-AUC primary).
  **No `remove_from_cart` in Oct** (docs claimed it — do not hardcode). 3,022,290
  users; category_id 0-null (reliable key) vs category_code 31.84% null. price
  0–2574.07, 0 negatives, 68,673 zeros. Time span full Oct in UTC.

**Both data cards COMPLETED** (docs/data_cards/). D-009 resolved (repo at C:\VAE,
734 GB free, not OneDrive). Data files confirmed gitignored (`data/*`, only
README tracked). No src/ code changed → no test re-run needed; scaffold suite
still green from Session 2.

**Phase 0 now at the GATE.** Awaiting owner sign-off (0.13). Open items for owner:
D-011-B license call, D-004 git identity check, D-005 (results/ gitignored — assent),
D-008 (GBT head — can defer to Phase 3). Nothing else blocks. Next phase after
sign-off = Phase 1 (Polars lazy preprocessing + leakage-safe splits + Nov download).

---

## 2026-07-09 — Session 4 (Phase 0 gate closed + Phase 1 executed)

Owner delegated the gate: *"Do what you think is best … ensure it is still
publishable."* → resolved D-004/005/008/011-B (D-012), then executed Phase 1.

**Environment fix (reproducibility):** the `.venv` was built in the old OneDrive
location and moved to C:\VAE, orphaning the editable install (`cadvae` unimportable,
`.pth` finder gone). Repaired with `uv pip install -e . --no-deps --python
.venv/Scripts/python.exe`. All Polars/pytest runs use `.venv/Scripts/python.exe`
(with `PYTHONPATH=src` for ad-hoc scripts) to avoid `uv run`'s editable-rebuild lock.

**Consequential decisions (owner-delegated, documented D-013..017, config-driven):**
A = stratified 60/20/20 on Response + train-only cleaning; B = temporal feature
window [Oct 1, Nov 22) → label [Nov 22, Nov 30], universe ≥5 events, user-disjoint
60/20/20 by hashed user_id, labels purchased/churned/next_category.

**Skeptical-numbers catch (CLAUDE.md §5):** the draft D-015 justified the label
window via a Black-Friday story. Real daily-purchase counts REFUTED it — the surge is
Nov 16–17 (185k on Nov 17), inside the feature window; Nov 29 is a mild 32k. Rationale
corrected; window kept (routine, unconfounded target). Universe threshold finalized
from a real sensitivity table (≥5 → 2.55M users / 3.27% positive).

**Built + verified:**
- A: `src/cadvae/data/personality.py` → engineered.parquet (2240×27) + `prepare()`
  (2240×34). Stratification preserved (0.149/0.150/0.150), no NaN, deterministic/seed.
- B: `src/cadvae/data/ecommerce.py` → user_table.parquet (2.55M×30, 78 MB) via one
  streaming pass (~200 s, bounded RAM) + `prepare()` (2.55M×27). Fixed a dead
  all-zero `cat_share_other` column (only 13 top-level cats exist < top_n=15).
- Leakage guards `tests/test_leakage.py` (8): perturbation proofs that train-fitted
  transforms ignore val/test (both datasets); label-window purchase excluded from
  features; temporal ordering; user-disjoint splits; real-cache cross-check.
- Integration `tests/test_pipeline.py` (2). **Full suite 25 passed; ruff + mypy clean.**
- `tests/test_config.py`: the D-007 null-guard was UPDATED (not weakened) to LOCK the
  approved split values (D-014/015) so they cannot drift silently.

**Phase 1 at the GATE (1.8).** Awaiting owner review of split logic + leakage tests +
data cards. All Phase-1 scientific choices are provisional/gate-reviewable and revert
via config + cache rebuild. Redundant raw `.zip`s deleted (717 GB free).

**Owner: "yes proceed" → Phase 2 executed same session.**

## 2026-07-09 — Session 4 cont. (Phase 2: constructs)

Named marketing constructs = the CA-DVAE alignment-head targets (interpretability
claim). Definitions D-018, encoded in `constructs:` config blocks; module
`src/cadvae/data/constructs.py`. Computed from the pre-standardization tables, split
with the SAME per-seed partition as `prepare()`, target matrix standardized
**train-only**; raw values returned for persona cards.
- **A (10 targets):** RFM (R=Recency, F=Σ Num*Purchases, M=Σ Mnt*), price_sensitivity
  = deal reliance (NumDeals/total ∈[0,1]), 6 spend-share affinities (sum to 1).
- **B (17 targets):** RFM (recency_days, n_purch, total_purch_value), price_sensitivity
  = −z(mean_view_price) price-tier proxy (honest limitation: no discount field), 13
  known-category shares. Affinity sum 0.69 ≡ the 68% non-null category_code rate — a
  clean internal-consistency check.
- Leakage guard `tests/test_constructs.py` (4): perturbation proof that val/test can't
  move the train-fitted construct scaler; split-alignment with prepare(); definition
  sanity. **Full suite 29 passed; ruff + mypy clean.**

**Phase 2 at the GATE (2.5).** Awaiting owner review of construct definitions.

**Owner: "proceed phase 3" → Phase 3 executed same session.**

## 2026-07-09 — Session 4 cont. (Phase 3: baselines / the bar to beat)

Frozen-representation protocol (D-019/020). Built: `models/ae.py` (AE),
`models/dec.py` (DEC, Xie 2016), `eval/metrics.py`, `eval/protocol.py` (RFM,
RFM+K-means, AE, AE+K-means, GMM-on-AE, DEC → 2 heads: L2-logistic + HistGBT),
`eval/stats.py` (mean±std + paired Wilcoxon), `eval/run_baselines.py` (Hydra
entrypoint → results/phase3/). Smoke test `tests/test_eval_smoke.py` (2) on tiny
CPU synthetic. K=8 clusters, latent d=16. **Full suite green; ruff + mypy clean.**

**Dataset A — the bar to beat (5 seeds, GPU, ~53 s), PR-AUC (base rate 0.149):**
| method | head | ROC-AUC | PR-AUC |
|---|---|---|---|
| **ae** | logreg | 0.857±0.007 | **0.572±0.042** |
| **ae** | gbt | 0.840±0.014 | **0.532±0.061** |
| gmm_ae | gbt | 0.763±0.020 | 0.399±0.051 |
| rfm | logreg | 0.763±0.012 | 0.394±0.015 |
| rfm | gbt | 0.725±0.032 | 0.349±0.045 |
| ae_kmeans / dec / rfm_kmeans | — | 0.70–0.74 | 0.30–0.36 |

→ **A bar to beat = AE embedding, PR-AUC ≈ 0.53 (gbt) / 0.57 (logreg).** Ordering
sane: continuous AE embedding > RFM(3) > K=8 cluster one-hots (clustering discards
signal). CA-DVAE must beat the AE embedding — a genuinely hard bar.

**Dataset B — 1-seed timing/sanity (subsample 200k, 40 epochs, ~5 min), PR-AUC
(base rate 0.033):** RFM gbt roc=0.789 pr=**0.195** (best) ≈ AE gbt roc=0.788
pr=0.175; clusters 0.07–0.10. **Honest finding: for behavioral data RFM ≈ AE** —
recency/frequency/monetary is a very strong purchase-prediction baseline.

**Bug + fix (integrity guard §3):** the first full 5-seed B run CRASHED — GMM-on-AE
hit singular covariance ("ill-defined empirical covariance / collapsed samples") on
a seed whose AE embedding had a near-collapsed latent dim (float32 + full covariance
→ singular). Fixed at the root (not by dropping the baseline): float64 embeddings +
`reg_covar=1e-3` + `covariance_type="diag"` (diag with a variance floor is
non-singular; standard for GMM-on-embeddings). Re-ran A to keep both datasets on the
identical protocol — A unchanged (gmm_ae 0.405 vs 0.399; bar still AE 0.532). B
relaunched. (Also fixed exit-code masking: `python ... | grep` had hidden the crash
as exit 0 — now the python exit code is captured directly.)

**Skeptical-numbers pass:** ordering plausible both datasets; all reps beat base
rate; ROC 0.79–0.86 (not too-good-to-be-true); temporal split clean → no leakage
flag; seed variance real. **Phase 3 at the GATE once the B run lands** — the bar to
beat is fixed BEFORE any CA-DVAE work (Phase 4).

## 2026-07-09 — Session 5 (Phase 3 B run landed + significance refinement → GATE)

Cold start: Phase 3 B run had landed while the last session was paused. Reconciled
provenance from result-dir mtimes, verified integrity, refined the stats, closed the
cheap checks, and brought Phase 3 to the gate.

**Dataset B — the bar to beat (5 seeds, GPU, subsample 200k / 40 epochs), PR-AUC
(base rate 0.033):**
| method | head | ROC-AUC | PR-AUC |
|---|---|---|---|
| **rfm** | gbt | 0.789±0.002 | **0.1947±0.0044** |
| ae | gbt | 0.786±0.004 | 0.1724±0.0056 |
| rfm | logreg | 0.769±0.003 | 0.1789±0.0033 |
| ae | logreg | 0.774±0.006 | 0.1625±0.0162 |
| gmm_ae | gbt | 0.741±0.013 | 0.1078±0.0080 |
| rfm_kmeans | gbt | 0.723±0.010 | 0.1085±0.0052 |
| dec | gbt | 0.706±0.017 | 0.0859±0.0050 |
| ae_kmeans | gbt | 0.642±0.007 | 0.0692±0.0057 |

→ **B bar to beat = RFM, gbt PR-AUC 0.1947.** RFM significantly beats AE
(paired-t p=3.6e-4). **Honest finding confirmed at 5 seeds: for behavioral data
RFM ≥ AE** — 3 raw recency/frequency/monetary features are a very strong
purchase-prediction baseline; the AE embedding of the full user×feature table does
not add lift. Ordering sane: RFM > AE > {gmm_ae ≈ rfm_kmeans} > dec > ae_kmeans; all
beat the 0.033 base rate; K-means hard one-hots are the weakest (assignment discards
signal). This is a genuinely hard, *non-neural* bar — good for the paper's honesty.

**Statistics refinement (D-019 updated; `paired_wilcoxon_vs_best` →
`significance_vs_best`).** Caught that a two-sided Wilcoxon signed-rank has a discrete
p-floor of 2^-(n-1) = **0.0625 at n=5** — it *cannot* reach p<0.05 no matter how
consistent the win (indeed every method's Wilcoxon p pins to exactly 0.0625 on both
datasets). CLAUDE.md permits Wilcoxon *or* paired t; we now report **both**, plus
`n_seeds` and the `wilcoxon_floor`, so the reader sees why the t-test carries the
5-seed claim. **Recommendation logged for Phase 5 headline claims: ≥6 seeds**
(Wilcoxon floor → 0.03125 < 0.05). Backward-compatible alias kept. Both baseline
`significance.json` recomputed from the saved per-seed `records.json`.

**Integrity checks (before touching anything):**
- Provenance from mtimes: A records written 19:32 @ commit `3bbd66e` (git_dirty=true
  — that dirtiness *was* the GMM fix, committed moments later as `0f889a1`); B records
  19:51 @ `0f889a1` clean. Both datasets therefore on the *identical fixed* protocol
  (robust diag-GMM). `significance.json` for both was patched in at 19:53 as a pure
  post-hoc recompute — records/summary untouched.
- **Verified** the patched `significance.json` reproduces byte-for-byte from
  `records.json` via the committed `significance_vs_best` (script:
  job tmp `verify_sig.py`) — for both datasets. So the stats layer is a faithful
  function of the real per-seed runs; no GPU re-run needed for a stats-only change.
- Removed the stale `wilcoxon.json` from both result dirs (superseded; the committed
  runner no longer emits it). `results/` is gitignored — local hygiene only.

**Cheap checks (CLAUDE.md §4):** `pytest` **31 passed** (+2 eval smoke incl. the
updated `test_aggregate_and_significance_over_seeds` asserting both families + the
n=2 Wilcoxon floor of 0.5); `ruff` clean; `mypy` clean (18 source files).

**Phase 3 is at the GATE (3.8).** The bar to beat is fixed and logged BEFORE any
CA-DVAE work: **A = AE embedding, PR-AUC 0.532 (gbt); B = RFM, PR-AUC 0.195 (gbt).**
Awaiting owner sign-off. Next phase after sign-off = Phase 4 (CA-DVAE + ablations).

## 2026-07-09 — Session 6 (Phase 4: CA-DVAE + ablations → GATE)

Owner: *"Implement next phase now."* → treated as the Phase-3 gate sign-off (bars
accepted) + authorization for Phase 4. Seed-count question (5 vs ≥6) carried to the
Phase-5 gate (it affects the sweep, not Phase 4).

**Built (model machinery, no separate ablation code):**
- `models/cadvae.py` — `CADVAE` (dense VAE, AE-matched backbone) + **linear**
  construct-alignment head over a designated aligned latent block; `cadvae_loss`
  (`recon + β·KL_free + w_a·KL_aligned + λ·align`, ELBO-scaled); `train_cadvae`
  (returns model + per-epoch loss history), `encode` (frozen posterior-mean μ),
  `align_predict`, `resolve_aligned_dims`. Design = **D-021** (alignment mechanism)
  + **D-022** (loss / disentanglement scope / ablations-as-config).
- `configs/model/cadvae.yaml` finalized; `aligned_dims: null`→auto rule.
- `eval/run_cadvae_sanity.py` — trains one point, logs curves, runs the *identical*
  Phase-3 frozen-rep heads, reports per-construct alignment R². Reuses
  `protocol._fit_eval_heads/_subsample` (same protocol → fair).
- `tests/test_cadvae_smoke.py` (6): train/shapes, determinism, both ablation config
  points, aligned/free KL split, `resolve_aligned_dims` rules. **Full suite 37 passed;
  ruff + mypy clean (20 files).**

**Sanity runs (single arbitrary point β=1, λ=1 — NOT the sweep; NOT tuned to bar):**
| dataset | aligned/free | downstream gbt PR-AUC | bar | Δ | align R² (mean / RFM) |
|---|---|---|---|---|---|
| A personality (200 ep) | 10/6 | 0.479 | AE 0.532 | −0.053 | 0.71 / **0.90** |
| B ecommerce (50k, 40 ep) | 12/4 | 0.150 | RFM 0.195 | −0.037 | 0.64 / 0.68 |

- **Alignment mechanism works** — the named axes genuinely encode the constructs
  (A: R=0.91/F=0.88/M=0.90; B RFM 0.68). This is direct evidence for the
  interpretability claim, independent of the lift claim.
- **CA-DVAE sits below both bars at this un-swept point** — exactly the predicted
  interpretability↔performance cost; the Phase-5 β/λ sweep maps the frontier.
  **Did NOT tune to close the gap** (integrity §3).
- **Skeptical-numbers observation to exploit in Phase 5:** `kl_free ≈ 0.02` on both
  datasets → at β=1 the free dims have **collapsed to the prior**; the aligned block
  carries the signal (kl_aligned 6.7 / 8.4). Sweeping β *down* (and/or λ) should
  reactivate free capacity and likely recover downstream PR-AUC — a concrete
  hypothesis for where the sweet spot lives. Not a bug (standard posterior collapse
  on unused dims); logged as a lead.
- **Real B timing anchor (promised):** 50k/40 ep end-to-end = **110 s** (incl. the
  2.55M-row parquet load); the 200k protocol run ≈ **3–4 min**, confirming the
  ~2–4 min/run estimate for the Phase-5 B sweep budget.

**Gotcha fixed:** the summary `print` used β/λ/²/→ → `UnicodeEncodeError` on the
Windows cp1252 console (job still wrote `sanity.json` first, so no data lost).
Rewrote `_print_summary` ASCII-only; both runs now exit 0. A's `metrics.jsonl`
(appended twice across the crash+fix) regenerated clean = 200 lines matching
`sanity.json`.

**Phase 4 is at the GATE (4.8).** Awaiting owner review of the model design (D-021/022)
+ sanity evidence. All choices config-driven + revertable. Next after sign-off =
Phase 5 (multi-seed × β/λ sweep) — **decide seed count (5 vs ≥6) at that gate.**

## 2026-07-09/10 - Session 7 (full repo audit; resumed after interruption)

Owner asked for a full repo audit before the Phase-4 gate review. The first audit
session made its fixes but was interrupted BEFORE documenting them (D-023..D-028
existed only as code-comment tags); this session verified the tree, finished the
audit, fixed what the interruption left broken, and wrote the documentation.

**Audit changes inherited from the interrupted session (verified this session):**
- Six seeds (D-023), multi-task downstream eval for B: churned + next_category on
  the same frozen representation; multiclass metrics (top-k acc, macro-OVR AUC).
- raw reference ceiling (excluded from the bar) + PCA-at-matched-capacity baseline
  + eval.methods subsetting (D-024).
- Tie-aware precision@k - deterministic, row-order invariant (D-025).
- NEW eval/interpretability.py: MIG / SAP / per-axis alignment with quantile-bin
  MI estimation + ground-truth-recovery tests (D-026). 6 tests.
- Bit-reproducibility (D-028): threadpool_limits(1) on representation-feeding
  sklearn fits; DEC distance via matmul expansion (cdist CUDA backward is
  nondeterministic); CPU randperm; user_id-sorted table; vocab name tiebreak.

**Broken state found on resume + fixed:**
- ecommerce.py next-category aggregation used filter() inside sort_by() keys ->
  polars STREAMING engine rejects it ("matching group lengths");
  test_B_label_window_purchase_excluded_from_features failed. Rewrote as ONE
  sorted column (when/otherwise NULL-mask -> sort_by(ts, product_id) ->
  drop_nulls().first()). Note: the tempting fix (sort values and mask separately)
  is WRONG - two multithreaded sorts need not agree on tied keys. Leakage suite
  8/8 green after fix.
- ruff E501 in _next_cat_codes signature.

**Audit continuation (previously un-audited modules):**
- models/cadvae.py + models/ae.py: loss/KL algebra verified correct (ELBO-scaled,
  per-dim KL, beta on free block only); seeded loaders OK. FOUND: silent CPU
  fallback in both trainers (contradicts CLAUDE.md hardware section + the config
  comment). Fixed via utils.device.resolve_device -> raises CudaUnavailableError
  (D-027). DEC inherits the AE device (covered).
- data/constructs.py, eval/run_cadvae_sanity.py, utils/{seeding,device,
  run_logging}.py, configs/*: audited, no defects. r2_rfm_mean's names[:3]==R/F/M
  assumption holds for both datasets (construct name order is fixed).
- eval/interpretability.py adversarial read: MIG normalization/gap correct
  (top1-top2)/H(v); SAP = squared Pearson (continuous form); degenerate factors
  NaN + excluded; deterministic (no RNG). Tests assert disentangled > Helmert-
  rotated code, exact axis recovery, bounds, JSON determinism.
- Known gap (expected, NOT a defect): stability metrics (ARI/NMI across seeds,
  bootstrap persistence) are Phase-5/6 work and do not exist yet. Nothing in
  src/ claims otherwise.

**Cheap checks (CLAUDE.md 4), full tree:** pytest 45 passed; ruff clean;
mypy clean (21 files). (uv trampoline briefly broke mid-session; checks were run
via .venv\Scripts\python.exe -m ... directly - environment quirk, not repo state.)

**Documentation debt cleared:** DECISIONS.md D-023..D-028 written (D-025/D-027
numbered at write-up; noted as such); CHECKLIST.md updated with the audit section.

**CONSEQUENTIAL - flagged for owner (blocks Phase 5):** the recorded Phase-3 bars
(A: AE gbt PR-AUC 0.532; B: RFM gbt 0.195) were produced under the OLD protocol -
5 seeds, non-tie-aware precision@k, nondeterministic DEC, no raw/pca methods, no
multi-task records. The audit changes the protocol (D-023/024/025/028), so the
bars are STALE and Phase 3 must be re-run (~1 min for A, ~30 min for B on the
4050) before any Phase-5 sweep. Re-run NOT launched autonomously: refreshing the
bar is gate-fixing evidence (owner sign-off pending at gates 1.8/2.5/3.8/4.8).

## 2026-07-10 - Session 7 (cont.): Phase-3 bar refresh under the audited protocol

Owner asked for a success-odds / publishing-angle assessment. The load-bearing
unknown is whether multi-task rescues the B story, so the stale bars are being
refreshed under the audited protocol (required before Phase 5 anyway). Old result
dirs preserved at results/phase3/*_baselines_5seed_preaudit.

**Dataset A re-run (6 seeds, raw+pca added, tie-aware prec@k):**
- Integrity check PASSED: per-seed PR-AUC for seeds 0-4 is bit-identical to the
  pre-audit run for AE and RFM (protocol changes did not perturb existing
  evidence); DEC differs slightly, exactly as expected from the D-028 rewrite.
- ERRATUM: the session-5 notebook entry quoted "AE gbt 0.532 +/- 0.006"; the
  true 5-seed std was +/- 0.061 (transcription slip; summary.json was correct).
- NEW BAR (gbt, pr_auc, 6 seeds): **pca = 0.5677 +/- 0.0729**, beating AE
  0.5156 +/- 0.0682 with paired-t p=0.0019 AND Wilcoxon p=0.03125 (both families
  significant at 6 seeds - D-023 vindicated). raw ceiling 0.6066; pca vs raw NOT
  significant (t p=0.19) -> at n=2,240 linear compression loses nothing; the
  neural AE is significantly WORSE than PCA. Honest finding, logged: the "hard
  neural baseline" is not the bar on A; PCA is. This RAISES the bar for CA-DVAE
  on A from 0.532 to 0.568.

**B re-run, attempt 1 (FAILED after 2.2h) + fix.** First relaunch omitted the
D-020 CLI overrides (eval.train_subsample=200000 model.max_epochs=40 - they live
in the run command, not the config), so it ran the FULL 1.53M train rows at 200
epochs. Crash exposed a REAL latent bug: >10k multiclass samples flip HistGBT
early_stopping='auto' ON, and its internal STRATIFIED validation split raises on
1-member classes (tail categories have them). Fix: early_stopping=False on the
multiclass head ONLY - at the D-020 subsample (~6.5k purchaser rows) auto is off
anyway, so the canonical protocol run is unchanged; binary heads untouched
(comparability). Regression test added (singleton class above the threshold).
Suite 46 passed, ruff+mypy clean. Lesson recorded: B runtime under D-028
single-threaded fits is ~5x the multithreaded 2026-07-09 runs - acceptable
one-off for Phase 3 (sweep does not refit baselines). Attempt 2 launched WITH
the D-020 overrides.

**B re-run, attempt 2 (SUCCESS, ~70 min, 6 seeds, D-020 overrides).** Integrity:
per-seed rfm/ae shift only +/-0.002 vs pre-audit - expected and explained: the
rebuilt cache is user_id-sorted (D-028), so the positional 200k subsample
differs; directionally identical. NEW BAR (primary, gbt): **rfm = 0.1953 +/-
0.0035**; every method below it with BOTH families significant (Wilcoxon 0.03125,
t<=1.5e-5); raw ceiling 0.2038 significantly ABOVE rfm -> non-RFM features carry
signal no learned rep currently captures.

**Multi-task results (the audit's D-023 payoff - CLAUDE.md tasks now all run):**
- churned (gbt pr_auc): ae 0.8620 best-of-methods, > rfm 0.8498 EVERY seed
  (t p=1.7e-8); raw 0.8676 slightly above ae (honest note).
- next_category: ae_kmeans acc@1 0.5550 stable across all 6 seeds, significantly
  > rfm 0.4070 (p=0.017), > raw 0.4520 (p=0.021), > pca (p=0.020). RFM collapses
  exactly as predicted (no category info). macro-OVR AUC: ae-family 0.63-0.66 vs
  rfm 0.52.
- SKEPTICAL FINDING (gold for the stability axis): plain AE embedding is BIMODAL
  on next_category across seeds (macroAUC 0.73/0.74/0.73 seeds 0-2 vs
  0.54/0.52/0.52 seeds 3-5; acc@1 0.58 vs 0.41-0.47) - the unregularized AE
  sometimes fails to allocate latent capacity to category shares. Direct,
  logged motivation for construct-anchored latents; Phase-6 ARI/NMI will
  quantify it. Also noted: several methods acc@1 < base_rate 0.5053 (heads
  optimize log-loss, not top-1; base_rate reported alongside).
- n_classes shows 12.67 in the summary = mean of per-seed integer class counts
  (aggregation artifact, records are exact).

**Phase-3 bars under the audited protocol are now FIXED: A = pca 0.5677 (gbt);
B = rfm 0.1953 (gbt).** Both runs: 6 seeds, tie-aware prec@k, deterministic
protocol, multi-task records logged. Old evidence preserved in
*_5seed_preaudit dirs.

## 2026-07-10 - Session 7 (cont.): gates signed off; Phase-5 sweep runner built

Owner signed off gates 1.8/2.5/3.8/4.8 (multi-task lift framing blessed at 3.8).
Audit checkpoint committed as `7b1361d` (clean tree BEFORE the sweep so every
Phase-5 run logs git_dirty=false).

**Pre-flight (before owner starts heavy compute):**
- Micro CA-DVAE runs vs the REBUILT B cache: A + B exit 0 (results/preflight).
- interpretability on real data: A instant; B 3.9 s at FULL 509k test rows;
  5/17 B factors NaN by design (near-zero-entropy rare-category shares, D-026).
- Disk 709 GB free. Suite 46 -> 49 passed after sweep tests; ruff+mypy clean.

**Phase 5 runner (`eval/run_cadvae_sweep.py`, D-029):** 6 beta x 4 lambda x 6
seeds = 144 runs/dataset; per-run JSON + model state_dict; resume-skip verified
(8/8 SKIP on relaunch, instant); ablations are grid rows/columns. Dry runs: A
2x2x2 grid 8/8 ok; B 2x2x2 on real cache 8/8 ok (multi-task heads + MIG/SAP per
run). **Protocol-scale timing anchor: one B run (200k/40ep) = 179.5 s -> B sweep
~7.5 h, A sweep ~1.5-2 h.** Encouraging single-point lead (NOT a result, not
tuned): at beta=1,lambda=1 full-scale B gives gbt pr_auc 0.1726 ~= AE bar-mate
0.1723 with MIG 0.306 - the 50k sanity gap to AE closes at 200k.

**Handoff to owner for overnight compute (exact commands):**
    powercfg /change standby-timeout-ac 0   # ONCE, before the night
    .venv\Scripts\python.exe -m cadvae.eval.run_cadvae_sweep data=ecommerce eval.train_subsample=200000 model.max_epochs=40
    .venv\Scripts\python.exe -m cadvae.eval.run_cadvae_sweep data=personality
Both are resumable: relaunching skips completed runs. Next after sweep = Phase 6
(stability ARI/NMI + bootstrap from saved models, trade-off curve, figures).

## 2026-07-10 - Session 7 (cont.): ALL remaining phase code implemented (no runs)

Owner: implement every phase''s code now, NO dry runs / compute; alterable after
the Phase-5 sweep lands. Interpretation of "no compute" per CLAUDE.md cheap-check
rule: unit tests on tiny synthetic data still run (seconds, CPU); no dataset
loads, no training runs, no sweeps.

**Built:**
- `eval/stability.py` (D-030) - rebuild_cadvae/load_model_state (weights_only),
  kmeans_assign (D-028 single-thread), pairwise ARI/NMI, bootstrap persistence,
  universe_matrix + fixed stability_sample.
- `eval/sweep_analysis.py` (D-031) - flatten/aggregate (NaN-aware), Pareto
  frontier, sweet-spot rule (slack = best point''s seed-std), per-seed extraction,
  paired Wilcoxon+t vs bar (zero-diff degenerate case -> p=1.0, not a crash).
- `viz/figures.py` (D-032) - trade-off curve (bar + ceiling lines, sweet-spot
  star, per-lambda series), stability bars, markdown tables, persona cards
  (latent traversal -> inverse-standardized raw-unit deltas).
- `eval/run_phase6.py` - orchestrator: consumes phase5 runs+models + phase3
  records only; emits analysis.json + figures/ + tables/; WARNS AND CONTINUES on
  partial sweeps (missing models/records -> logged warning, not a crash).
- `eval/repro_check.py` (Phase 7) - re-runs one pinned seed of the baseline
  protocol, diffs every numeric metric vs recorded records.json (NaN==NaN),
  exit 1 on mismatch; `repro:` config block (atol for cross-machine).
- configs: `phase6:` + `repro:` blocks. README rewritten as the Phase-7
  reproduction guide (all commands, recorded evidence table).

**Crash-proofing evidence (the owner''s explicit ask):** 16 new tests incl. an
END-TO-END run of analyze() against a fully faked sweep+baseline directory tree
(exact artifact shapes the real sweep writes), guard tests for every raise path
(k>n, mismatched lengths, no-finite-points, lambda=0 cards), degenerate-stats
paths, and a torch.load weights_only round-trip. Full suite **65 passed**;
ruff + mypy clean (27 files). Known deliberate limits: run_phase6 on REAL sweep
output is still untested by definition (no compute allowed) - first real
invocation after the sweep is the integration test, and D-030..032 are expected
to be tuned then (owner pre-authorized).

**Hygiene note for Phase 7:** `threadpoolctl` is imported directly but is only a
transitive dep (via scikit-learn) in pyproject - add as direct dependency at the
next `uv lock` (needs network; deferred deliberately).

## 2026-07-10 - Session 7 (cont.): pre-compute audit of Phases 5/6/7 (D-033)

Owner asked for a Phase-5/6/7 audit before starting compute, with
publication-improving changes allowed. Five findings, all fixed and tested:

1. **CRITICAL - sweet spot was selected on TEST** (sweep recorded no val
   metrics). Now: val metrics per run, selection on val, reporting on test
   (D-033), drift-guard test pins the dual-eval heads to the Phase-3 spec.
   Caught BEFORE the sweep - records would otherwise have needed regeneration.
2. Per-task significance used the primary-bar method (rfm) for every task -
   straw man on next_category where ae_kmeans is the real bar. Now: best
   PER-TASK baseline + a multitask.md comparison table (the blessed claim''s
   table).
3. Clustering metrics (CLAUDE.md protocol) were never computed for CA-DVAE.
   Now: silhouette/DB/CH per stability config (mean over seeds).
4. Sweep record writes were not atomic -> a power cut could leave a corrupt
   file that resume counts as DONE and Phase 6 crashes on. Now: tmp+replace;
   corrupt records raise naming the file + recovery step.
5. Phase-6 stability re-ran prepare() per config x seed (18 B-parquet loads)
   and embedded before checking model files. Now: per-seed universe memoization
   + file pre-check.

Checks: full suite **68 passed** (4 new tests: dual-head drift guard, per-task
baseline winner, corrupt-record naming, e2e assertions for val-selection +
clustering + multitask table); ruff + mypy clean. B sweep cost +~10-15 s/run
(val eval) -> ~8 h overnight; A unchanged ~2 h. Launch commands unchanged
(README). Publication posture: strictly improved - the two most likely
methodology objections (test-set tuning, straw-man baselines) are now dead
before any compute is spent.
