# Data directory

Nothing under `data/` is committed except this README. Raw files, interim
artifacts, and processed Parquet caches are all gitignored.

```
data/
  raw/
    customer_personality/marketing_campaign.csv          (Dataset A, ~2,240 rows)
    ecommerce_multicategory/2019-Oct.csv                 (Dataset B, ~42M events, ~5.7 GB)
    ecommerce_multicategory/2019-Nov.csv                 (Dataset B, ~67M events, ~9 GB)
  interim/      intermediate artifacts (Phase 1+)
  processed/    aggregated user×feature Parquet caches (Phase 1+)
```

## Dataset A — Customer Personality Analysis

- Kaggle: <https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis>
- Download `marketing_campaign.csv` and place it at
  `data/raw/customer_personality/marketing_campaign.csv`

## Dataset B — eCommerce behavior data (multi-category store)

- Kaggle: <https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store>
- Download `2019-Oct.csv` and `2019-Nov.csv` and place them under
  `data/raw/ecommerce_multicategory/`
- These are multi-GB files. If this repo lives inside a OneDrive-synced folder,
  pause OneDrive sync (or exclude/relocate the repo) before placing them.

## Kaggle CLI — verified working procedure (2026-07-09, CLI 2.2.3)

Auth uses a **Kaggle API token** (`KGAT_…`, from kaggle.com → Settings → API →
Create New Token) via the `KAGGLE_API_TOKEN` env var. `kaggle` is **not** a project
dependency — run it in an isolated `uv tool` env so it never touches `.venv`:

```bash
# token stored at ~/.kaggle/access_token (outside the repo, chmod 600)
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"

# Dataset A (~220 KB) — --unzip works for full-dataset downloads
python -m uv tool run --from kaggle kaggle datasets download \
  -d imakash3011/customer-personality-analysis \
  -p data/raw/customer_personality --unzip

# Dataset B — download per-file with -f (each is multi-GB).
# GOTCHA: with -f, --unzip is a no-op — the CLI leaves a .zip you must extract:
python -m uv tool run --from kaggle kaggle datasets download \
  -d mkechinov/ecommerce-behavior-data-from-multi-category-store \
  -f 2019-Oct.csv -p data/raw/ecommerce_multicategory
unzip -o data/raw/ecommerce_multicategory/2019-Oct.csv.zip \
  -d data/raw/ecommerce_multicategory   # → 2019-Oct.csv (5.67 GB)
# repeat with -f 2019-Nov.csv for the second month (9.01 GB) at Phase 1 start.
```

Licenses reported by the CLI at download: Dataset A = **CC0-1.0**; Dataset B =
**`copyright-authors`** (© original authors — restrictive, not an open license;
owner review required — see DECISIONS.md D-011).
