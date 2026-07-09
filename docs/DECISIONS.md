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
