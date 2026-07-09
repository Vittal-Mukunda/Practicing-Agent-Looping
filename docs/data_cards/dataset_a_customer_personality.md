# Data card — Dataset A: Customer Personality Analysis

**STATUS: COMPLETED from first-hand inspection on 2026-07-09.** Schema, dtypes,
row counts, missingness, ranges, and quirks below were emitted by `polars`
inspection of the actual file (not from public documentation). Split /
missingness / outlier policy remain PENDING — those are Phase 1 gate decisions.

## Provenance
- Source: Kaggle, `imakash3011/customer-personality-analysis`
  <https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis>
- Original provider credited on page: Dr. Omar Romero-Hernandez (unverified)
- **License: CC0-1.0 (public domain)** — confirmed first-hand 2026-07-09; the
  Kaggle CLI reported `License(s): CC0-1.0` at download. Resolves the Dataset-A
  half of D-011.
- Retrieved: 2026-07-09 via `kaggle datasets download` (CLI 2.2.3).
- File: `data/raw/customer_personality/marketing_campaign.csv` (220,188 bytes)
- SHA256: `cd0affa36b1b981e80ba0e27767e9b3ab723f7a3dba948f722af81abc6b990ea`

## Role in the study
Interpretability story: rich demographics + spending + campaign-response columns.
Stratified splits (mandated, CLAUDE.md). Primary downstream task: campaign
response (`Response`).

## Verified shape
**2,240 rows × 29 columns.** Single file, **tab-separated** (`\t`; 28 tabs in the
header, 0 commas — confirmed). 2,241 physical lines (1 header + 2,240 data).

## Schema (verified via polars)
Legend: dtype inferred by polars; `null` = null count; `uniq` = distinct values.

| # | Column | dtype | null | uniq | range / notes |
|---|--------|-------|-----:|-----:|---------------|
| 1 | ID | Int64 | 0 | 2240 | customer key, 0–11191 (not contiguous) |
| 2 | Year_Birth | Int64 | 0 | 59 | 1893–1996; **3 rows < 1920** (1893, 1899, 1900) — implausible ages |
| 3 | Education | String | 0 | 5 | Graduation, Basic, 2n Cycle, PhD, Master |
| 4 | Marital_Status | String | 0 | 8 | **junk levels present:** Absurd (2), Alone (3), YOLO (2) alongside Married/Single/Together/Divorced/Widow |
| 5 | Income | Int64 | **24** | 1975 | 1,730–**666,666**; p99 = 94,472 → the max is a lone extreme outlier; 24 nulls |
| 6 | Kidhome | Int64 | 0 | 3 | 0–2 |
| 7 | Teenhome | Int64 | 0 | 3 | 0–2 |
| 8 | Dt_Customer | String | 0 | 663 | enrollment date, **DD-MM-YYYY** string — needs explicit format parse |
| 9 | Recency | Int64 | 0 | 100 | 0–99 days since last purchase (**R** in RFM) |
| 10 | MntWines | Int64 | 0 | 776 | 0–1493 (monetary) |
| 11 | MntFruits | Int64 | 0 | 158 | 0–199 |
| 12 | MntMeatProducts | Int64 | 0 | 558 | 0–1725 |
| 13 | MntFishProducts | Int64 | 0 | 182 | 0–259 |
| 14 | MntSweetProducts | Int64 | 0 | 177 | 0–263 |
| 15 | MntGoldProds | Int64 | 0 | 213 | 0–362 |
| 16 | NumDealsPurchases | Int64 | 0 | 15 | 0–15 |
| 17 | NumWebPurchases | Int64 | 0 | 15 | 0–27 |
| 18 | NumCatalogPurchases | Int64 | 0 | 14 | 0–28 |
| 19 | NumStorePurchases | Int64 | 0 | 14 | 0–13 |
| 20 | NumWebVisitsMonth | Int64 | 0 | 16 | 0–20 |
| 21 | AcceptedCmp3 | Int64 | 0 | 2 | 0/1, pos rate 0.073 |
| 22 | AcceptedCmp4 | Int64 | 0 | 2 | 0/1, pos rate 0.075 |
| 23 | AcceptedCmp5 | Int64 | 0 | 2 | 0/1, pos rate 0.073 |
| 24 | AcceptedCmp1 | Int64 | 0 | 2 | 0/1, pos rate 0.064 |
| 25 | AcceptedCmp2 | Int64 | 0 | 2 | 0/1, pos rate **0.013** (very rare) |
| 26 | Complain | Int64 | 0 | 2 | 0/1, pos rate 0.009 |
| 27 | Z_CostContact | Int64 | 0 | **1** | constant = 3 → **no information, drop** |
| 28 | Z_Revenue | Int64 | 0 | **1** | constant = 11 → **no information, drop** |
| 29 | Response | Int64 | 0 | 2 | **primary label** (last campaign) |

## Primary label balance
`Response`: n = 2,240, positives = **334**, negatives = 1,906, **positive rate =
0.149**. Class-imbalanced → report **PR-AUC** alongside ROC-AUC; stratify splits
on this label unless the gate decides otherwise.

## Construct-relevant columns (for Phase 2 planning, not yet fitted)
- **RFM:** R = `Recency`; F = sum of `Num*Purchases`; M = sum of `Mnt*` columns.
- **Price sensitivity:** `NumDealsPurchases`, deal share vs. total purchases.
- **Category affinity:** the six `Mnt*` product families (wines / fruits / meat /
  fish / sweets / gold).
(Exact construct definitions are a **Phase 2 gate** decision — consequential.)

## Data-quality issues found (to resolve at Phase 1, consequential)
1. `Income`: 24 nulls + one 666,666 outlier (≫ p99 94,472). Imputation/winsor
   policy is a gate decision.
2. `Year_Birth`: 3 implausible rows (<1920). Drop vs. cap is a gate decision.
3. `Marital_Status`: 7 rows in junk levels (Absurd/Alone/YOLO). Recode/drop is a
   gate decision.
4. `Z_CostContact`, `Z_Revenue`: constant — drop before modeling (uncontroversial).
5. `Dt_Customer`: parse DD-MM-YYYY explicitly (do not rely on locale inference).

## Splits, missingness policy, outlier policy
PENDING — Phase 1 gate decisions (consequential; owner sign-off required). Configs
hold `null` for `test_size` / `val_size` / `stratify_on` until then (D-007).
