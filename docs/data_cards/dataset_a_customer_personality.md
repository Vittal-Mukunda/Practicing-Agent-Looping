# Data card — Dataset A: Customer Personality Analysis

**STATUS: SKELETON — the file is not on disk yet, so nothing below has been
verified against the actual data. Every field marked (unverified) comes from the
dataset's public documentation, not from inspection. This card is completed —
schema, dtypes, row counts, missingness, ranges, quirks — the moment
`data/raw/customer_personality/marketing_campaign.csv` exists. Do not model
against this card until the PENDING sections are filled and the Phase 0 gate is
signed off.**

## Provenance
- Source: Kaggle, `imakash3011/customer-personality-analysis`
  <https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis>
- Original provider credited on page: Dr. Omar Romero-Hernandez (unverified)
- License: PENDING owner confirmation on the Kaggle page (commonly labeled CC0 — unverified; D-011)
- Retrieved: PENDING (record download date + file checksum here)

## Role in the study
Interpretability story: rich demographics + spending + campaign-response columns.
Stratified splits (mandated, CLAUDE.md). Downstream task: campaign response.

## Expected shape (unverified)
~2,240 rows × 29 columns; single CSV, reportedly **tab-separated** — verify
delimiter on first inspection.

## Schema
PENDING — to be emitted from actual `polars` inspection (names, dtypes, null
counts, min/max, cardinalities). Do **not** assume column names (CLAUDE.md).

## Known-quirk candidates to check on inspection (from public discussion — unverified)
- Delimiter is tab, not comma.
- `Income` (or equivalent) has nulls and at least one extreme outlier.
- Year-of-birth column implies implausible ages (people born <1920).
- Two constant columns (`Z_CostContact`, `Z_Revenue` or similar) carry no information.
- Date column (`Dt_Customer` or similar) needs explicit format parsing.

## Splits, missingness policy, outlier policy
PENDING — Phase 1 gate decisions (consequential; owner sign-off required).
