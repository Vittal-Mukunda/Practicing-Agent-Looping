"""Dataset A (Customer Personality Analysis) preprocessing.

Leakage-safe pipeline (CLAUDE.md frozen-representation + no-leakage protocol):

1. ``engineer`` — row-wise feature engineering that uses NO cross-row information
   (Age, tenure, junk-category collapse, column drops). Seed-independent, so it is
   computed once and cached to Parquet.
2. ``prepare`` — per-seed **stratified** split (D-014), then fit imputation /
   winsorization / one-hot / standardization on the **train split only** (D-013)
   and apply the fitted transforms to val/test without refitting.

The train-only fitting is the whole point: a leakage-guard test asserts that the
scaler/encoder/winsor bounds depend only on train rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Column roles after engineering (verified against the real schema — data card A).
NUMERIC_COLS = [
    "Income", "Age", "Kidhome", "Teenhome", "Recency",
    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds",
    "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumWebVisitsMonth", "Complain", "Customer_Tenure_Days",
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
]
CATEGORICAL_COLS = ["Education", "Marital_Status"]
WINSOR_COLS = ["Income", "Age"]  # only these carry the known extreme outliers


def engineer(cfg: DictConfig) -> pl.DataFrame:
    """Row-wise, seed-independent feature engineering (no leakage possible)."""
    pp = cfg.preprocess
    df = pl.read_csv(cfg.raw_path, separator=cfg.separator)

    df = df.with_columns(
        (pl.lit(pp.reference_year) - pl.col("Year_Birth")).alias("Age"),
        (
            pl.lit(pp.tenure_reference).str.to_date("%Y-%m-%d")
            - pl.col("Dt_Customer").str.to_date(pp.dt_customer_format)
        ).dt.total_days().alias("Customer_Tenure_Days"),
        pl.when(pl.col("Marital_Status").is_in(list(pp.marital_status_junk)))
        .then(pl.lit(pp.marital_status_other))
        .otherwise(pl.col("Marital_Status"))
        .alias("Marital_Status"),
    )
    keep = ["ID", cfg.label, *NUMERIC_COLS, *CATEGORICAL_COLS]
    return df.select(keep)


def cache_engineered(cfg: DictConfig) -> Path:
    """Write the engineered (pre-split, leakage-free) frame to Parquet once."""
    out_dir = Path(cfg.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "engineered.parquet"
    engineer(cfg).write_parquet(path)
    return path


@dataclass
class PreparedSplit:
    """Frozen, leakage-safe train/val/test matrices for one seed."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    ids: dict[str, np.ndarray]  # split -> customer ID array (traceability)
    # fitted, train-only transforms (kept for persona-card inverse mapping, Phase 6)
    fitted: dict = field(default_factory=dict)


def _stratified_indices(y: np.ndarray, cfg: DictConfig, seed: int):
    """Two-stage stratified split → train / val / test index arrays."""
    test_size = cfg.split.test_size
    val_size = cfg.split.val_size
    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed
    )
    # val fraction expressed relative to the remaining train+val pool
    val_rel = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_rel, stratify=y[train_val_idx], random_state=seed
    )
    return train_idx, val_idx, test_idx


def prepare(cfg: DictConfig, seed: int, engineered: pl.DataFrame | None = None) -> PreparedSplit:
    """Stratified split + TRAIN-ONLY fit of all statistical transforms."""
    if engineered is None:
        path = Path(cfg.processed_dir) / "engineered.parquet"
        engineered = pl.read_parquet(path) if path.exists() else engineer(cfg)

    y = engineered[cfg.label].to_numpy()
    ids_all = engineered["ID"].to_numpy()
    num = engineered.select(NUMERIC_COLS).to_numpy().astype(np.float64)
    cat = engineered.select(CATEGORICAL_COLS).to_numpy()

    tr, va, te = _stratified_indices(y, cfg, seed)

    # --- impute (train median; only Income has nulls) ---
    col_idx = {c: i for i, c in enumerate(NUMERIC_COLS)}
    medians = np.nanmedian(num[tr], axis=0)
    inds = np.where(np.isnan(num))
    num[inds] = np.take(medians, inds[1])

    # --- winsorize Income, Age to TRAIN quantiles ---
    bounds = {}
    for c in WINSOR_COLS:
        j = col_idx[c]
        lo = np.quantile(num[tr, j], cfg.preprocess.winsor_lower_q)
        hi = np.quantile(num[tr, j], cfg.preprocess.winsor_upper_q)
        num[:, j] = np.clip(num[:, j], lo, hi)
        bounds[c] = (float(lo), float(hi))

    # --- standardize (fit on train) ---
    scaler = StandardScaler().fit(num[tr])
    num_z = scaler.transform(num)

    # --- one-hot categoricals (fit on train; unseen val/test levels -> all-zero) ---
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cat[tr])
    cat_oh = enc.transform(cat)
    cat_names = list(enc.get_feature_names_out(CATEGORICAL_COLS))

    X = np.hstack([num_z, cat_oh])
    feature_names = list(NUMERIC_COLS) + cat_names

    return PreparedSplit(
        X_train=X[tr], X_val=X[va], X_test=X[te],
        y_train=y[tr].astype(np.int64), y_val=y[va].astype(np.int64), y_test=y[te].astype(np.int64),
        feature_names=feature_names,
        ids={"train": ids_all[tr], "val": ids_all[va], "test": ids_all[te]},
        fitted={"medians": medians, "winsor_bounds": bounds, "scaler": scaler, "encoder": enc},
    )
