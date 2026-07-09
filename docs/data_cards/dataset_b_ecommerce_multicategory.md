# Data card — Dataset B: eCommerce behavior, multi-category store

**STATUS: COMPLETED for `2019-Oct.csv` from first-hand streaming inspection on
2026-07-09** (`polars.scan_csv` + `engine="streaming"`; 5.67 GB read with bounded
RAM, ~59 s). `2019-Nov.csv` is **not yet downloaded** — it shares this schema
(same store, next month) and is deferred to Phase 1 start; its per-file counts /
time range / user-overlap are marked PENDING below. Split boundaries and label
definitions remain Phase 1 gate decisions.

## Provenance
- Source: Kaggle, `mkechinov/ecommerce-behavior-data-from-multi-category-store`
  <https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store>
- Data provider credited on page: REES46 Marketing Platform.
- **License/terms — ACTION REQUIRED.** Kaggle CLI reported `License(s):
  copyright-authors` at download — i.e. "Data files © Original Authors", a
  **restrictive** tag, *not* an open/CC license and *not* the softer "free for
  research" wording on the HF mirror. Owner must review the Kaggle page's license
  section and decide acceptability for a public paper + code release (see
  DECISIONS.md **D-011**; Phase 0 gate item). Attribution to REES46 + the Kaggle
  page will be included regardless.
- Retrieved: 2026-07-09, `2019-Oct.csv` via `kaggle datasets download -f`
  (CLI 2.2.3), then `unzip` (the `-f` download ignores `--unzip`).
- File: `data/raw/ecommerce_multicategory/2019-Oct.csv` (5,668,612,855 bytes).

## Role in the study
Scale + downstream-lift story. **Temporal** splits (mandated, CLAUDE.md; the #1
leakage surface). Downstream tasks: next-purchase-category / repeat purchase;
churn/dormancy (engineered). Multi-category store chosen over the cosmetics shop
so the category-affinity construct is non-vacuous (D-006).

## Verified schema — 2019-Oct.csv (9 columns)
Comma-separated. **42,448,764 rows** (Oct only). Each row = one user event.

| # | Column | dtype | null (Oct) | notes |
|---|--------|-------|-----------:|-------|
| 1 | event_time | String | 0 | `YYYY-MM-DD HH:MM:SS UTC` string — **parse explicitly** (UTC); do not rely on locale inference |
| 2 | event_type | String | 0 | **only `view` / `cart` / `purchase` in Oct** — see note below |
| 3 | product_id | Int64 | 0 | 166,794 distinct |
| 4 | category_id | Int64 | 0 | **624 distinct, zero nulls → the reliable category key** |
| 5 | category_code | String | **13,515,609 (31.84%)** | 127 distinct; human-readable taxonomy but **~1/3 missing** |
| 6 | brand | String | 6,113,008 (14.40%) | 3,446 distinct |
| 7 | price | Float64 | 0 | min 0.0, max 2574.07, mean 290.32; **0 negatives**, 68,673 exact-zero |
| 8 | user_id | Int64 | 0 | **3,022,290 distinct users** (Oct) |
| 9 | user_session | String | 2 | 9,244,422 distinct sessions; 2 null rows |

## event_time range (Oct)
min `2019-10-01 00:00:00 UTC` → max `2019-10-31 23:59:59 UTC` (full calendar
month, UTC). This is the temporal space the Phase 1 split boundaries carve up.

## event_type distribution (Oct) — matters for label prevalence
| event_type | count | share |
|---|---:|---:|
| view | 40,779,399 | 96.07% |
| cart | 926,516 | 2.18% |
| purchase | 742,849 | 1.75% |

**Purchase rate ≈ 1.75%** → severe class imbalance for purchase-based downstream
labels → **PR-AUC is the primary metric**, ROC-AUC secondary; report top-k.

## Findings that override the "documented" schema (CLAUDE.md: never assume)
1. **`remove_from_cart` is ABSENT in October.** Public docs / the skeleton listed
   view/cart/**remove_from_cart**/purchase; the Oct file contains only three types.
   Verify per-month before writing any code that branches on `remove_from_cart`;
   do not hardcode its existence. (May appear in Nov — check at Phase 1.)
2. **`category_code` is 31.84% null**, so the category-affinity construct should
   key on **`category_id`** (0 nulls) and treat `category_code` as an optional
   human-readable label. Consequential for Phase 2 — flagged, not decided.
3. `brand` 14.40% null; `price` has 68,673 exact-zero rows (free items /
   placeholders?) worth a policy decision but no negatives.

## Facts still PENDING (Nov file + gate decisions)
- `2019-Nov.csv`: per-file row count, event_time range, event_type mix, and
  **Oct↔Nov user overlap** (drives churn/repeat labels + temporal split) — download
  at Phase 1 start.
- Duplicate-event strategy: documented, not yet decided.
- Aggregation plan: user×feature matrix via Polars lazy `group_by` → Parquet cache
  once (3.02M users Oct; 16 GB RAM → never materialize raw frame). Phase 1.

## Splits (temporal boundaries), label definitions
PENDING — Phase 1 gate decisions (consequential; the #1 leakage surface; owner
sign-off required). Configs hold `null` for `train_end`/`val_end`/`test_end` and
`files` until then (D-007).
