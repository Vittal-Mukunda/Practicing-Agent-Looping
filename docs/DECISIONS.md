# DECISIONS.md — every non-obvious choice + rationale

Format: `D-NNN (date) — decision — rationale — status`.
Status: **locked** (approved / uncontroversial engineering), **provisional** (needs
gate sign-off), **pending** (awaiting owner input).

---

**D-001 (2026-07-09) — `uv` as environment + lockfile manager.**
One tool covers Python-version pinning (`.python-version`), dependency locking
(`uv.lock`, committed), and venv management; invoked as `python -m uv` so it works
without PATH changes. Alternatives (poetry, pip-tools) rejected for no functional
gain and slower resolution. *Status: locked (engineering).*

**D-002 (2026-07-09) — torch pinned to 2.6.0 from the cu124 index.**
Exact pin in `pyproject.toml`; wheel `2.6.0+cu124` (Windows, cp312) via
`download.pytorch.org/whl/cu124`. cu124 chosen over cu126/cu128 for wider driver
tolerance; RTX 4050 (Ada, compute capability 8.9) is fully supported by every
CUDA 12.x build. Verified working by `python -m cadvae.utils.device` (evidence in
NOTEBOOK.md). torchvision/torchaudio omitted — tabular project, not needed.
*Status: locked (engineering).*

**D-003 (2026-07-09) — uv resolution restricted to `sys_platform == 'win32'`.**
The paper's numbers are produced on one fixed machine (Windows / RTX 4050). A
single-platform lockfile is smaller and cannot silently resolve different versions
on other OSes. Cross-OS reproduction is future work; the committed lockfile is the
authoritative environment. *Status: locked (engineering).*

**D-004 (2026-07-09) — repo-local git identity set to "Vittal Muku" <vittal.muku@gmail.com>.**
No global git identity existed; commits require one. Name derived from the email
(matches the session's authenticated account). Owner delegated gate decisions
2026-07-09; identity kept as-is — owner may still override any time via
`git config user.name "..."`. *Status: locked (owner-delegated).*

**D-005 (2026-07-09) — `results/` is gitignored.**
Per-run outputs (checkpoints, metrics, resolved configs) are bulky and regenerable
from committed code + configs + seeds; that is the reproducibility contract.
Curated result tables/figures get promoted into `docs/` (or a tracked
`results/summary/`) deliberately at analysis time (Phase 6). *Status: locked (owner-delegated 2026-07-09).*

**D-006 (2026-07-09) — Dataset B = multi-category store (not the cosmetics shop).**
Owner decision from the kickoff message: the category-affinity construct is
near-vacuous with a single-category store; the multi-category set also carries the
scale story. *Status: locked (owner).*

**D-007 (2026-07-09) — null-means-undecided convention for consequential config values.**
Split ratios/boundaries, stratification column, and aligned-dims are consequential
(CLAUDE.md §6) and await their phase gates. They are explicit `null` in configs —
never silently defaulted — and `tests/test_config.py::test_undecided_phase1_parameters_are_null_not_defaults`
enforces it. Phase 1+ code must refuse to run on nulls. *Status: locked (protocol).*

**D-008 (2026-07-09) — GBT downstream head: sklearn `HistGradientBoostingClassifier`.**
Zero extra dependency, strong tabular default, CPU-fast, deterministic under a fixed
`random_state`. XGBoost/LightGBM would add a dep without an obvious accuracy story at
these data sizes. Metric-affecting → consequential, but the choice is standard and
identical across every model/baseline (it is the *evaluation* head, applied uniformly),
so it does not bias the comparison. Locked now under owner delegation for a fixed
protocol; revisitable at Phase 3 if evidence warrants. *Status: locked (owner-delegated 2026-07-09).*

**D-009 (2026-07-09) — repo location risk: OneDrive-synced folder. RESOLVED.**
Original concern: repo under `C:\Users\vitta\OneDrive\Desktop\VAE` would sync
`.venv` (~5 GB) and `data/raw` (~15 GB). **Resolved 2026-07-09: repo relocated to
`C:\VAE` (non-synced; confirmed `pwd` = `/c/VAE`).** 734 GB free on C:, ample for
the ~15 GB Dataset B. No OneDrive interference. *Status: locked (resolved).*

**D-010 (2026-07-09) — determinism posture.**
`seed_everything` seeds python/numpy/torch(+CUDA), forces deterministic cuDNN,
disables cudnn.benchmark, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and enables
`torch.use_deterministic_algorithms(True, warn_only=True)` so nondeterministic ops
are *surfaced* in logs rather than silently allowed (CLAUDE.md: document unavoidable
GPU nondeterminism). Revisit warn-only vs strict at Phase 4 when the training loop
exists. *Status: locked (engineering), revisit Phase 4.*

**D-011 (2026-07-09) — dataset licenses, now verified first-hand at download.**
Kaggle pages are JS-rendered, so earlier programmatic verification failed; the
Kaggle **CLI**, however, prints the license string at download time.
- **Dataset A → `CC0-1.0` (public domain).** Confirmed first-hand. No restriction;
  no attribution legally required (we will credit the source anyway). RESOLVED.
- **Dataset B → `copyright-authors`** (Kaggle's "Data files © Original Authors"
  label). This is the **actual Kaggle license tag** and it CONTRADICTS the softer
  HF-mirror wording ("free to use for research… please mention the source"). It is
  *not* an open/CC license. Practically, academic-research use of REES46's public
  Kaggle release with attribution is the common and intended use, but the label is
  restrictive, not permissive.
  **DECISION (2026-07-09, owner-delegated) — proceed with Dataset B under a
  publishable posture:**
  1. **No redistribution.** Raw CSVs *and* derived Parquet caches stay gitignored;
     the reproduction package instructs readers to download from Kaggle themselves.
     "© Original Authors" forbids copying their files — we never do; we publish only
     derived aggregate statistics + code that points to the original source.
  2. **Attribution.** REES46 Marketing Platform + the Kaggle URL cited prominently
     in README and paper (data/acknowledgements section).
  3. **Non-commercial academic research use only.**
  Rationale/publishability: this REES46 release is used in hundreds of peer-reviewed
  papers under exactly this arrangement (attribution + non-redistribution); the
  posture is standard and defensible to a skeptical reviewer. Dataset A (CC0) carries
  the interpretability story independently, so the work is not license-fragile. If a
  target venue ever demands a permissive data license, the eCommerce *scale* result
  is still fully reproducible by any reader from the Kaggle source. Residual risk
  (REES46 revokes public access) is low and owner-accepted.
*Status: A resolved (CC0); B resolved (proceed, posture above).*

**D-012 (2026-07-09) — Phase 0 gate: PASSED (owner-delegated sign-off).**
Owner response at the gate: *"Do what you think is best and still has a high
quality output and ensure that it is still publishable."* Interpreted as authority
to resolve the open Phase 0 decisions (D-004/005/008/011-B) and advance, subject to
the standing constraints of publication-grade rigor and honest evaluation. All
gate artifacts complete: repo structure + Hydra schema (committed scaffold), both
data cards filled from first-hand inspection, licenses resolved. Advancing to
Phase 1. **Consequential Phase-1 scientific decisions (split boundaries/ratios,
stratification column, label definitions, feature definitions) will still be made
explicitly and documented here with rationale as new D-entries, and remain
config-driven and revertable** — the delegation authorizes forward progress, not
silent or irreversible scientific choices. *Status: locked.*

---
## Phase 1 consequential decisions (owner-delegated; reviewable at Phase 1 gate)

All fits below are **train-only** and applied to val/test without refitting
(CLAUDE.md frozen-representation + no-leakage protocol). Every parameter lives in
the data configs; changing any is a config edit + Phase-1 cache rebuild (cheap
relative to Phases 3-6), so these remain revertable at the gate.

**D-013 (2026-07-09) — Dataset A feature set + cleaning policy.**
*Features (VAE/AE input, per customer):* `Income`(cleaned), `Age`(=2014−Year_Birth,
cleaned), `Kidhome`, `Teenhome`, `Recency`, the 6 `Mnt*`, the 4 `Num*Purchases`,
`NumWebVisitsMonth`, `Complain`, `Customer_Tenure_Days`(from `Dt_Customer` vs fixed
reference 2015-01-01), one-hot `Education`(5), one-hot `Marital_Status`(cleaned),
and prior-campaign flags `AcceptedCmp1..5`.
*Why AcceptedCmp1..5 are features, not leakage:* the label `Response` is the **6th
(last)** campaign; Cmp1..5 are strictly-earlier campaigns → legitimate past→present
predictors. *Dropped:* `ID`, `Year_Birth`(→Age), `Dt_Customer`(→tenure),
`Z_CostContact`/`Z_Revenue`(constant), `Response`(label).
*Cleaning (train-fit):* `Income` nulls → train median; `Income` & `Age` winsorized to
[train p1, train p99] (absorbs the 666,666 income and the <1920 births); junk
`Marital_Status` {Absurd,YOLO,Alone}→"Other"; unseen categorical levels in val/test →
all-zero one-hot; all numeric features z-scored on train mean/std. Reference year
2014 chosen because `Dt_Customer` spans 2012–2014 (campaign era). *Status: provisional (gate-reviewable).*

**D-014 (2026-07-09) — Dataset A split = stratified 60/20/20 on `Response`.**
2,240 rows, 14.9% positive → 20% test ≈ 448 rows ≈ 67 positives (usable ROC/PR-AUC);
val 20% for downstream-head tuning. Partition re-drawn per seed (≥5 seeds → mean±std
+ Wilcoxon). Stratifying on the label balances prevalence across folds and is **not**
leakage — the label is used only to partition; representation learning never sees it.
Config: `test_size=0.2, val_size=0.2, stratify_on=Response`. *Status: provisional (gate-reviewable).*

**D-015 (2026-07-09) — Dataset B temporal windows + user universe (THE #1 leakage surface).**
Single temporal feature→label separation:
- **Feature window** `[2019-10-01 00:00:00Z, 2019-11-21 23:59:59Z]` (52 days of history).
- **Label window** `[2019-11-22 00:00:00Z, 2019-11-30 23:59:59Z]` (9 days).
- **Universe:** users with **≥5 events** in the feature window → **2.55M users**,
  label-window purchase rate **3.27%** (finalized from real data — see below).
- **Guarantee:** every feature event strictly precedes every label event → no temporal
  leakage; a leakage-guard test asserts `max(feature event_time) < label_window_start`.
- **Train/val/test:** users partitioned **disjointly** 60/20/20 by a deterministic hash
  of `user_id` (seed-salted) → no user appears in two splits (blocks identity leakage).
  Representation + construct targets fit on **train users only**.

*Verification against the real 110M-row data (2026-07-09) — and a CORRECTED
rationale.* The original draft justified this window by a "predict Black-Friday
purchasing" story. **That was wrong and the data refuted it** (caught by the
skeptical-numbers pass, CLAUDE.md §5): the purchase surge is **Nov 16–17**
(185k purchases on Nov 17, ~7× the ~24k/day baseline), which falls **inside the
feature window**; Black Friday (Nov 29) shows only a mild bump (32k). The label
window Nov 22–30 is therefore a **routine, promotion-unconfounded** future period —
which is actually a *cleaner* downstream target ("predict routine future purchasing
from 52 days of history, including any prior promo behavior") than a sale-confounded
one. Window kept; rationale corrected.

*Universe threshold — chosen from the observed sensitivity table (min feature-window
events → users / label-window positive rate):* ≥1 → 4.74M / 2.00%; ≥2 → 3.74M /
2.44%; ≥3 → 3.22M / 2.75%; **≥5 → 2.55M / 3.27%**; ≥10 → 1.78M / 4.08%. Chose **≥5**:
buyer personas (RFM, category affinity, price sensitivity) are meaningless for a
1–2-view drive-by user, so a minimum behavioral footprint is required; ≥5 retains
scale (2.55M users) while excluding near-zero-signal traffic. Config-driven
(`universe.min_events_feature_window`) → trivially swept as a robustness check.
Class imbalance (~3.3% positive) → **PR-AUC primary**. Config `train_end/val_end/
test_end` repurposed as split anchors.
*Status: provisional (gate-reviewable); universe threshold now finalized from data.*

**D-016 (2026-07-09) — Dataset B label definitions (from the label window).**
1. **`purchased` (primary, binary):** ≥1 `purchase` event in the label window. Drives
   the primary downstream-lift claim.
2. **`next_category` (multiclass / top-k):** top-level category (from `category_code`,
   mapped via a `category_id`→top-level lookup built from non-null rows to recover the
   31.84% missing) of the user's **first** label-window purchase; evaluated top-k over
   the most frequent categories; defined only for label-window purchasers.
3. **`churned` (binary, dormancy proxy):** active in feature window but **zero** events
   of any type in the label window. Short-horizon proxy (2-month data) — limitation
   noted for the paper; longer-horizon churn is future work.
All label-window events are excluded from features by construction (D-015). *Status: provisional (gate-reviewable).*

**D-017 (2026-07-09) — Dataset B per-user feature aggregation (representation input).**
Aggregated over the feature window via Polars lazy `scan_csv→filter→group_by→
collect(engine="streaming")` (never materialize the ~110M-row raw frame; 16 GB RAM):
- **Recency:** days from user's last feature-window event to label-window start.
- **Frequency:** n_events, n_views, n_carts, n_purchases, n_sessions, n_active_days.
- **Monetary:** total/mean/max purchase `price`; total/mean cart `price`.
- **Category affinity:** event-share over the top-N top-level categories (+`other`) —
  the category-affinity construct input (N set at Phase 2; features cached wide).
- **Price/behavior:** mean_view_price, mean_purchase_price, cart_abandon_rate
  (1−purchases/carts, div-0 guarded), conversion=purchases/views (guarded).
Output = one row per universe user → Parquet cache (loaded small for every experiment).
*Status: provisional (gate-reviewable).*

---
## Phase 2 constructs (owner-delegated; reviewable at Phase 2 gate)

**D-018 (2026-07-09) — Named marketing constructs = the alignment-head targets.**
These are the *named* axes the CA-DVAE aligns latent dims to (interpretability
claim). Each is a continuous target; the full target matrix is **standardized on the
TRAIN split only** (leakage-safe — a guard test proves val/test cannot move the
fitted stats). Raw (unstandardized) values are also returned for persona-card
interpretation (Phase 6). Constructs are computed from the pre-standardization
tables (engineered.parquet / user_table.parquet), so they inherit the exact same
per-seed split as `prepare()`.

*Dataset A (Customer Personality):* 10 targets.
- **RFM (3):** R = `Recency`; F = Σ of the 4 `Num*Purchases`; M = Σ of the 6 `Mnt*`.
- **Price sensitivity (1):** `deal_reliance` = `NumDealsPurchases` / max(F, 1) — a
  genuine discount-seeking measure (A records deal purchases). Higher = more sensitive.
- **Category affinity (6):** spend share `Mnt_c / max(M, 1)` for wines/fruits/meat/
  fish/sweets/gold.

*Dataset B (eCommerce):* 17 targets.
- **RFM (3):** R = `recency_days`; F = `n_purch` (feature-window purchases — sparse by
  nature, 77% of users have 0, which faithfully reflects non-buyers); M =
  `total_purch_value`.
- **Price sensitivity (1):** `view_price_tier` = −z(`mean_view_price`) — a **price-tier
  proxy** (budget vs premium browser), defined for every user. **Honest limitation:**
  the event log has no discount field, so unlike A this is *not* discount-responsiveness
  but the price level a user engages with; documented as such for the paper.
- **Category affinity (13):** the 13 known top-level `cat_share_*` columns (excludes
  `unknown`; it is not a named construct).

Orientation of every target is documented in code. Definitions live in the data
configs (`constructs:` block) → swappable/reviewable. *Status: provisional (gate-reviewable).*

---
## Phase 3 baselines / evaluation protocol (owner-delegated; reviewable at gate)

**D-019 (2026-07-09) — frozen-representation downstream protocol.**
Each method yields a per-user frozen representation → two downstream heads (D-008:
L2-logistic + `HistGradientBoostingClassifier`) trained on **train**, evaluated on
**test**. Representations (all fit on train only, per seed):
- **RFM (incumbent floor, continuous):** the 3 standardized RFM construct values.
- **RFM+K-means:** hard cluster one-hot (K) from K-means on train RFM.
- **AE (the hard baseline, continuous):** the d-dim AE embedding.
- **AE+K-means:** hard cluster one-hot (K) on the frozen AE embedding.
- **GMM-on-AE:** soft responsibilities (K) from a GMM on the frozen AE embedding
  (diagonal covariance, `reg_covar=1e-3`, float64 — robust to near-collapsed AE dims;
  full covariance was singular on some seeds; see NOTEBOOK fix).
- **DEC:** soft cluster assignment (K).
(β-VAE / VAE+align / CA-DVAE embeddings are added in Phase 4 — same protocol.)
Reporting continuous *and* segment representations answers CLAUDE.md's requirement
that every listed baseline gets a downstream-lift number, and shows how much
predictive signal clustering discards vs the raw embedding.
- **Metrics:** ROC-AUC + **PR-AUC (primary; both labels imbalanced)** + top-k.
  Clustering methods also get silhouette / Davies–Bouldin / Calinski–Harabasz and
  (Phase 6) ARI/NMI stability.
- **Statistics:** mean ± std over **≥5 seeds** (seeds = 0..4), + paired significance
  vs the strongest baseline. **We report BOTH paired Wilcoxon AND paired t** because
  two-sided Wilcoxon signed-rank has a discrete p-floor of 2^-(n-1) = **0.0625 at
  n=5** — it *cannot* reach p<0.05 regardless of effect size, so at 5 seeds the
  t-test carries the significance claim. **Recommendation for Phase 5 headline claims:
  ≥6 seeds** (Wilcoxon floor → 0.03125). The "bar to beat" = best baseline test
  PR-AUC, fixed and logged BEFORE any CA-DVAE run (Phase 3 gate).
  *Status: provisional (gate-reviewable).*

**D-020 (2026-07-09) — latent dim, cluster count, Dataset B training subsample.**
- **Latent dim d = 16** (CLAUDE.md 8–32; balanced default; config `model.latent_dim`).
- **K = 8 clusters** for all segmentation baselines (comparable across methods; config
  `eval.n_clusters`). Silhouette-based K selection noted as a robustness alternative.
- **Dataset B training subsample:** representation learning (AE/DEC/K-means/GMM fit)
  and head fitting use a seed-stratified sample of **train** users
  (`eval.train_subsample`, default 200k) for tractability on the 6 GB / 16 GB laptop
  across many runs; **evaluation is always on the FULL held-out test set** (no
  subsampling of test). Documented; the subsample size is config-driven and its
  effect is a logged robustness check.
- **Measured + finalized (2026-07-09):** Dataset B uses `eval.train_subsample=200000`
  + `model.max_epochs=40` → ~5 min/seed (~25 min for the 5-seed run) on the RTX 4050.
  Dataset A (2,240 rows) runs full (no subsample, 200 epochs) in ~53 s / 5 seeds.
  Reproduction commands are logged in each run's `resolved_config.yaml` and in the
  NOTEBOOK. *Status: provisional (gate-reviewable).*

**D-021 (2026-07-09) — CA-DVAE construct-alignment mechanism (Component 1).**
- **Designated aligned block.** The latent vector is split `z = [z_aligned (a) | z_free (d−a)]`.
  A **linear** head `A: R^a → R^C` predicts the C standardized Phase-2 construct targets
  (RFM / price-sensitivity / category-affinity) from the aligned block; alignment loss =
  summed-MSE. **Linear (not MLP) is deliberate:** each construct then corresponds to an
  interpretable *direction* in latent space and the per-axis alignment score (Phase 6, MIG/SAP)
  is well defined. A non-linear head would re-introduce the black box we are trying to remove.
- **`aligned_dims` (a) auto-rule.** `a = min(n_constructs, latent_dim − min_free_dims)` with
  `min_free_dims = 4`, so ≥4 free dims always remain for the disentanglement block. Resolves to
  **A: a=10 (=n_constructs; 6 free)**, **B: a=12 (17 correlated construct targets, 4 free)**.
  B's 13 category-affinity shares are compositional/low-rank, so a 12-dim block predicting 17
  targets is not a bottleneck (measured test alignment R² mean 0.64, RFM 0.68). Config
  `model.aligned_dims` (null→auto) + `model.min_free_dims`; both gate-reviewable.
- **Frozen embedding = posterior mean μ (all d dims)** for the downstream protocol — identical
  treatment to the AE baseline (which uses its encoder output), so lift differences are about
  the *objective*, not the read-out. *Status: provisional (gate-reviewable).*

**D-022 (2026-07-09) — CA-DVAE loss, disentanglement scope, ablation parameterization.**
- **Loss:** `L = recon_MSE + β·KL_free + w_a·KL_aligned + λ·align_MSE` (CLAUDE.md
  `recon + β·KL + λ·align`, made explicit). **β weights only the FREE-dim KL** (CLAUDE.md:
  "β-VAE KL weighting on the remaining free dims"); aligned dims keep a standard unit-Gaussian
  prior at weight `w_a = aligned_kl_weight = 1.0` so they stay stochastic instead of collapsing
  under alignment pressure.
- **ELBO reduction convention:** recon and each KL block are **summed over their dimension,
  averaged over the batch** (align likewise over constructs) — the standard VAE ELBO scaling, so
  β, λ ≈ O(1) are meaningful. Absolute scale is absorbed by the Phase-5 sweep regardless.
- **Ablations are pure config points (no separate code paths):** β-VAE = `λ=0` (→ `a` forced 0,
  β weights the whole KL — a *pure* β-VAE); VAE+align = `β=1`; CA-DVAE full = `β,λ` swept. This
  is the "two independently ablatable components" requirement satisfied by one model class.
- **β, λ are the Phase-5 swept variables** that generate the interpretability–performance
  trade-off curve. *Status: provisional (gate-reviewable).*

---
*The following entries (D-023…D-028) were implemented during the full-repo audit
(2026-07-09/10). The audit session was interrupted before writing them down; they
were reconstructed from the code and completed on 2026-07-10. D-025 and D-027 were
numbered at write-up time.*

**D-023 (2026-07-10) — six seeds + multi-task downstream evaluation.**
- **`eval.seeds = [0..5]` (6 seeds).** Resolves the open D-019 recommendation: the
  two-sided Wilcoxon p-floor 2^-(n-1) becomes 0.03125 < 0.05, so BOTH paired test
  families (Wilcoxon + t) can carry a headline significance claim.
- **Multi-task frozen-rep protocol (implements CLAUDE.md "Tasks" fully).** Each
  method's representation is learned ONCE per seed; only the downstream heads are
  refit per task. Records carry a `task` field (`primary` = the dataset's main
  label). Dataset B extra tasks (labels engineered in D-016): **churned** (binary
  dormancy proxy = no label-window event) and **next_category** (multiclass =
  top-level category of the FIRST label-window purchase, time order with product_id
  tiebreak). Dataset A has only `Response` (its `labels` dict is empty — interface
  parity).
- **next_category class space** = the feature-window category vocabulary
  (+ `other` for label-window categories outside it, + `unknown` for undecodable) —
  temporally safe: no label/test information defines the classes. The label is
  decoded via a static product-taxonomy lookup (`category_id → top_cat`, recovering
  the 31.84%-null `category_code` events); the lookup feeds ONLY the label, never a
  feature, so no temporal restriction applies.
- **Multiclass metrics:** top-k accuracy (k = 1/3/5, config `data.labels.next_category.top_k`)
  + macro one-vs-rest ROC-AUC over classes present in test. *Status: provisional
  (gate-reviewable).*

**D-024 (2026-07-10) — reference ceiling, PCA baseline, method subsetting.**
- **`raw`** (full standardized feature vector, no representation learning) is run
  through the same heads as a **reference ceiling**: it answers the reviewer question
  "do you even need a representation?". It is **excluded from bar/best-baseline
  selection** (`exclude_from_best=("raw",)` in `significance_vs_best`) but is still
  compared against the bar like every other method.
- **`pca`** at matched capacity (d = `model.latent_dim`, full SVD) joins the baseline
  set — the classical linear-compression comparison the reviewer will ask for.
- **`eval.methods`** (null = all) selects a subset of baselines for partial runs;
  unknown names raise. DEC mutates the shared AE encoder in place, so it always runs
  LAST within the AE family. *Status: provisional (gate-reviewable).*

**D-025 (2026-07-10) — tie-aware precision@k.**
- Tree heads and cluster one-hot representations emit piecewise-constant scores, so
  exact ties at the top-k boundary are common; naive argsort top-k then depends on
  on-disk row order (silently arbitrary, not reproducible). `precision_at_k` now
  scores boundary ties by their **expected positive rate for the remaining slots**
  (the average over all tie-breaking orders) — deterministic, row-order invariant,
  and equal to plain top-k precision whenever there are no boundary ties.
  *Status: provisional (gate-reviewable — it is a metric-definition change).*

**D-026 (2026-07-10) — interpretability estimator conventions (MIG/SAP/alignment).**
- `eval/interpretability.py`: **MIG** (Chen et al., NeurIPS 2018, eq. 6), **SAP**
  (Kumar et al., ICLR 2018, continuous-factor form = squared Pearson), and the
  per-construct **axis-alignment score** (best single-dim R² + winning dim — the
  persona-card pointer). Ground-truth factors = the Phase-2 standardized construct
  targets (that IS the claim under test).
- Estimator conventions follow Locatello et al. (ICML 2019) / disentanglement_lib:
  MI on discretized variables, 20 bins. **Documented deviation:** our factors are
  continuous, so BOTH latents and factors use **quantile (equal-mass) bins** —
  robust to the heavy tails of monetary/count features (equal-width bins are not).
  (Near-)zero-entropy factors are reported NaN and excluded from means. Everything
  is deterministic (no RNG). Ground-truth-recovery unit tests: a disentangled code
  must outscore a Helmert-rotated (entangled) code of identical information content.
  *Status: provisional (gate-reviewable).*

**D-027 (2026-07-10) — no silent CPU fallback in trainers.**
- `train_autoencoder` and `train_cadvae` previously substituted CPU when CUDA was
  requested but unavailable — exactly the silent fallback CLAUDE.md's hardware
  section forbids. Both now use `utils.device.resolve_device`, which raises
  `CudaUnavailableError`; CPU must be requested explicitly (`device=cpu`, as the
  tests do). DEC inherits the AE's device, so it is covered transitively.

**D-028 (2026-07-10) — bit-reproducibility policy (sklearn fits + DEC).**
- Identity re-runs (same seed, same config, twice) exposed DEC as the single
  non-bit-reproducible baseline. Three root causes, all fixed:
  (1) **multithreaded-BLAS reduction order** makes sklearn fits wobble ~1e-7 across
  bit-identical inputs — absorbed by discrete outputs, but DEC uses KMeans centers
  as CONTINUOUS init and training amplifies the wobble chaotically. Every sklearn
  fit that feeds a representation (KMeans/GMM/PCA) now runs under
  `threadpool_limits(1)`; heads stay multithreaded (measured bit-identical).
  (2) **`torch.cdist` CUDA backward** uses nondeterministic atomicAdd → DEC's
  soft-assign distance is now the matmul expansion (deterministic under
  `CUBLAS_WORKSPACE_CONFIG`), clamped at 0.
  (3) **CUDA `randperm`** under warn-only deterministic mode → batch permutations
  are drawn on CPU and moved.
- **Deterministic row order everywhere positional operations happen:** the Dataset-B
  user table is sorted by `user_id` at build AND at load (streaming `group_by` emits
  arbitrary order), and the category vocabulary breaks frequency ties by name.

**D-029 (2026-07-10) — Phase-5 sweep grid + runner design.**
- **Grid:** `beta in {0.05, 0.1, 0.25, 0.5, 1.0, 2.0} x lambda_align in
  {0, 0.25, 1.0, 4.0}` x 6 seeds = 144 runs/dataset. beta is swept DOWN from 1
  because both Phase-4 sanity runs showed `kl_free ~ 0.02` at beta=1 (free dims
  collapsed to the prior); the lambda=0 column IS the pure beta-VAE ablation and
  the beta=1 row IS the VAE+align ablation (D-022 config points, no extra runs).
  Config `sweep:` block; gate-reviewable at Phase 5.
- **Runner (`eval/run_cadvae_sweep.py`), deliberate deviations from naive Hydra
  multirun:** internal grid loop so `prepare()` is paid once per SEED (not per
  run — ~24x saving on B's parquet+constructs cost); per-(seed,beta,lambda)
  JSON with resume-skip (a mid-night crash costs one ~3-min run, not the batch);
  model state_dicts saved (~120 KB each) so Phase 6 computes clustering/stability
  (ARI/NMI, bootstrap) for SELECTED configs without retraining — per-run KMeans
  deliberately excluded from the sweep (most expensive CPU step under D-028
  single-threading; only needed for configs Phase 6 inspects).
- **Interpretability factors** = the Phase-2 standardized construct targets on
  the TEST split (Z = posterior-mean embedding of test); logged per run alongside
  downstream metrics for all tasks — the two axes of the headline curve.
- **Measured at protocol scale (RTX 4050):** B run (200k/40ep) = 179.5 s ->
  B sweep ~7.5 h (one overnight batch); A sweep ~1.5-2 h. *Status: provisional
  (gate-reviewable).*

**D-030 (2026-07-10) — stability protocol (Phase 6).**
- **Cross-seed:** a fixed, seed-INDEPENDENT sample of the user universe
  (`phase6.stability_users`, RNG `phase6.stability_seed` — not a model seed) is
  embedded by each seed''s saved sweep model (each seed''s own train-only scaler:
  the FULL per-seed pipeline is what must be stable), K-means partitioned
  (K = eval.n_clusters, single-threaded per D-028), scored by mean pairwise ARI
  (Hubert & Arabie 1985) + NMI (Strehl & Ghosh 2002) — both permutation-invariant,
  so cluster relabeling across seeds does not depress the score. Universe row
  alignment across seeds is asserted, not assumed.
- **Bootstrap persistence:** refit on bootstrap resamples, ARI of full-population
  assignments vs the reference fit (cf. von Luxburg 2010), `phase6.n_boot` = 20.
- **Configs compared:** the D-031 sweet spot + the two D-022 ablation points —
  all from SAVED sweep models (no retraining). The plain-AE instability evidence
  comes from the logged Phase-3 multi-task records (2026-07-10 bimodality finding).
  *Status: provisional (gate-reviewable; may be revised after the sweep — owner
  authorized post-sweep alteration 2026-07-10).*

**D-031 (2026-07-10) — sweet-spot selection rule (trade-off curve).**
- Sweet spot = among Pareto-frontier points of (interpretability x, downstream y),
  the highest-x point whose y-mean is within `y_slack` of the best y-mean anywhere
  on the grid; default slack = the best point''s own seed-std ("statistically
  indistinguishable from the best"). Fallback = best-y point. Rule inputs + chosen
  point are logged in analysis.json — the choice is reported, not hidden.
  x = MIG by default (`phase6.x_metric`); the alignment-R2 curve variant is always
  emitted alongside. *Status: provisional (gate-reviewable).*

**D-032 (2026-07-10) — persona cards + figure conventions.**
- Persona card = latent traversal of ONE aligned axis from -span to +span (prior
  units, `phase6.traversal_span`=2), decoded and inverse-standardized to raw
  feature units; top-N |delta| features shown per axis; axes titled by their
  best-aligned construct (from the run''s axis_alignment). Cards use the
  sweet-spot model at the lowest seed.
- Figures: matplotlib Agg only, PNG 200 dpi, error bars = seed std, no style
  packages (reviewer reproducibility). *Status: provisional (gate-reviewable).*
