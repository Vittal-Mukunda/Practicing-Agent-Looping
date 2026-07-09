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

## Kaggle CLI alternative

With `%USERPROFILE%\.kaggle\kaggle.json` configured (kaggle.com → Settings → API → Create New Token):

```powershell
python -m uv tool run kaggle datasets download -d imakash3011/customer-personality-analysis -p data/raw/customer_personality --unzip
python -m uv tool run kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store -p data/raw/ecommerce_multicategory --unzip
```

License terms for both datasets must be confirmed on their Kaggle pages before
publication use (Phase 0 gate item).
