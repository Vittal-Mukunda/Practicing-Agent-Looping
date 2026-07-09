# Data card — Dataset B: eCommerce behavior, multi-category store

**STATUS: SKELETON — files not on disk yet; nothing verified against actual data.
This card is completed from real `polars` lazy scans (schema, per-file row counts,
event-time ranges, event-type distribution, null profile) once the CSVs exist
under `data/raw/ecommerce_multicategory/`. Do not model against this card until
the PENDING sections are filled and the Phase 0 gate is signed off.**

## Provenance
- Source: Kaggle, `mkechinov/ecommerce-behavior-data-from-multi-category-store`
  <https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store>
- Data provider credited on page: REES46 Marketing Platform
- License/terms: PENDING owner confirmation on the Kaggle page. Secondary source
  (HF mirror card) states: "free to use for research, books, and educational
  materials. Please mention the source" — attribution to REES46 + the Kaggle page
  will be included in the paper regardless (D-011).
- Retrieved: PENDING (record download date + file checksums here)

## Role in the study
Scale + downstream-lift story. Temporal splits (mandated, CLAUDE.md). Downstream
tasks: next-purchase-category / repeat purchase; churn/dormancy (engineered).
Multi-category store chosen over cosmetics shop because the category-affinity
construct requires multiple categories (D-006).

## Expected shape (unverified)
Two CSVs, `2019-Oct.csv` (~42M events, ~5.7 GB) and `2019-Nov.csv` (~67M events,
~9 GB). Documented columns (verify — do not assume): `event_time`, `event_type`
(view / cart / remove_from_cart / purchase), `product_id`, `category_id`,
`category_code`, `brand`, `price`, `user_id`, `user_session`.

## Schema
PENDING — from `polars.scan_csv` (streaming; 16 GB RAM constraint — never load whole).

## Facts to establish on inspection
- Exact per-file row counts; `event_time` min/max per file (defines the temporal-split space).
- Event-type value counts (purchase rate matters for label prevalence / PR-AUC).
- Null profile: `category_code`, `brand`, `user_session` expected to have nulls (unverified).
- Duplicate-event check strategy (documented, not assumed).
- User overlap between Oct and Nov (relevant to split design and churn labels).

## Splits (temporal boundaries), label definitions
PENDING — Phase 1 gate decisions (consequential; the #1 leakage surface; owner sign-off required).
