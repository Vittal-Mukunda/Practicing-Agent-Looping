"""Dataset B (eCommerce multi-category) preprocessing.

Scale + downstream-lift story. The raw event log is ~110M rows / ~15 GB, so every
aggregation is a Polars **streaming** lazy scan — the raw frame is NEVER
materialized (16 GB RAM limit, CLAUDE.md hardware section).

Leakage-safe design (D-015/016/017):

* **Feature window** [feature_window_start, label_window_start) → per-user features.
* **Label window** [label_window_start, label_window_end] → per-user labels.
  Every feature event strictly precedes every label event (temporal guarantee;
  a leakage-guard test asserts it).
* **Universe** = users with >= ``min_events_feature_window`` feature-window events.
* The per-user table is seed-independent → aggregated **once** and cached to Parquet.
* **Train/val/test** are user-disjoint, assigned by a seed-salted hash of ``user_id``
  at load time (no user in two splits → blocks identity leakage). Standardization is
  fit on **train users only**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

CATEGORY_PREFIX = "cat_share_"


def _parse_anchor(s: str) -> datetime:
    """Parse a UTC ISO anchor like '2019-10-01T00:00:00Z' to a naive datetime."""
    return datetime.fromisoformat(s.replace("Z", ""))


def _scan(cfg: DictConfig) -> pl.LazyFrame:
    files = [str(Path(cfg.raw_dir) / f) for f in cfg.files]
    return pl.scan_csv(files).with_columns(
        pl.col("event_time").str.slice(0, 19).str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ts"),
        pl.col("category_code").str.split(".").list.first().alias("top_cat"),
    )


def _category_vocab(cfg: DictConfig, fs: datetime, ls: datetime) -> list[str]:
    """Top-N top-level categories by FEATURE-WINDOW frequency (temporally safe:
    no label-window information enters the feature vocabulary)."""
    n = int(cfg.features.top_n_categories)
    vocab = (
        _scan(cfg)
        .filter((pl.col("ts") >= fs) & (pl.col("ts") < ls) & pl.col("top_cat").is_not_null())
        .group_by("top_cat")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(n)
        .collect(engine="streaming")
    )
    return vocab["top_cat"].to_list()


def build_user_table(cfg: DictConfig) -> pl.DataFrame:
    """One streaming pass → per-user features + labels, universe-filtered."""
    fs = _parse_anchor(cfg.split.feature_window_start)
    ls = _parse_anchor(cfg.split.label_window_start)
    le = _parse_anchor(cfg.split.label_window_end)
    vocab = _category_vocab(cfg, fs, ls)

    scan = _scan(cfg).with_columns(
        ((pl.col("ts") >= fs) & (pl.col("ts") < ls)).alias("in_feat"),
        ((pl.col("ts") >= ls) & (pl.col("ts") <= le)).alias("in_label"),
        (pl.col("event_type") == "view").alias("is_view"),
        (pl.col("event_type") == "cart").alias("is_cart"),
        (pl.col("event_type") == "purchase").alias("is_purch"),
    )

    _purch_price = pl.col("price").filter(pl.col("in_feat") & pl.col("is_purch"))
    aggs = [
        pl.col("in_feat").sum().alias("n_events"),
        (pl.col("in_feat") & pl.col("is_view")).sum().alias("n_views"),
        (pl.col("in_feat") & pl.col("is_cart")).sum().alias("n_carts"),
        (pl.col("in_feat") & pl.col("is_purch")).sum().alias("n_purch"),
        pl.col("user_session").filter(pl.col("in_feat")).n_unique().alias("n_sessions"),
        pl.col("ts").filter(pl.col("in_feat")).dt.date().n_unique().alias("n_active_days"),
        pl.col("ts").filter(pl.col("in_feat")).max().alias("last_ts"),
        _purch_price.sum().alias("total_purch_value"),
        _purch_price.mean().alias("mean_purch_value"),
        _purch_price.max().alias("max_purch_value"),
        pl.col("price").filter(pl.col("in_feat") & pl.col("is_view")).mean()
        .alias("mean_view_price"),
        # labels (label window)
        (pl.col("in_label") & pl.col("is_purch")).any().alias("label_purchase"),
        pl.col("in_label").any().alias("label_any_event"),
    ]
    # per-category feature-window event counts (raw; converted to shares below)
    for c in vocab:
        aggs.append((pl.col("in_feat") & (pl.col("top_cat") == c)).sum().alias(f"__cnt_{c}"))
    aggs.append((pl.col("in_feat") & pl.col("top_cat").is_null()).sum().alias("__cnt_unknown"))

    tbl = scan.group_by("user_id").agg(aggs).filter(
        pl.col("n_events") >= int(cfg.universe.min_events_feature_window)
    ).collect(engine="streaming")

    # --- derive shares + ratios (vectorized, on the small aggregated table) ---
    cnt_cols = [f"__cnt_{c}" for c in vocab] + ["__cnt_unknown"]
    known_sum = pl.sum_horizontal([pl.col(c) for c in cnt_cols])
    share_names = [f"{CATEGORY_PREFIX}{c}" for c in vocab] + [f"{CATEGORY_PREFIX}unknown"]
    n_ev = pl.col("n_events")
    share_exprs = [
        (pl.col(f"__cnt_{c}") / n_ev).alias(f"{CATEGORY_PREFIX}{c}") for c in vocab
    ]
    share_exprs.append((pl.col("__cnt_unknown") / n_ev).alias(f"{CATEGORY_PREFIX}unknown"))
    # "other" (categories beyond the top-N) only exists if the vocab was truncated;
    # with the full vocabulary captured it is structurally 0 → omit the dead feature.
    if len(vocab) >= int(cfg.features.top_n_categories):
        share_exprs.append(((n_ev - known_sum) / n_ev).alias(f"{CATEGORY_PREFIX}other"))
        share_names.append(f"{CATEGORY_PREFIX}other")

    tbl = tbl.with_columns(
        # recency: days from user's last feature event to the label window start
        (ls - pl.col("last_ts")).dt.total_seconds().truediv(86400.0).alias("recency_days"),
        pl.when(pl.col("n_carts") > 0)
        .then(1.0 - pl.col("n_purch") / pl.col("n_carts"))
        .otherwise(0.0).alias("cart_abandon_rate"),
        pl.when(pl.col("n_views") > 0)
        .then(pl.col("n_purch") / pl.col("n_views"))
        .otherwise(0.0).alias("conversion"),
        *share_exprs,
    ).with_columns(
        # no-purchase / no-view users: value/price means are null → 0 (n_purch disambiguates)
        pl.col("total_purch_value").fill_null(0.0),
        pl.col("mean_purch_value").fill_null(0.0),
        pl.col("max_purch_value").fill_null(0.0),
        pl.col("mean_view_price").fill_null(0.0),
    )

    base = [
        "recency_days", "n_events", "n_views", "n_carts", "n_purch",
        "n_sessions", "n_active_days", "total_purch_value", "mean_purch_value",
        "max_purch_value", "mean_view_price", "cart_abandon_rate", "conversion",
    ]
    keep = ["user_id", *base, *share_names, "label_purchase", "label_any_event"]
    return tbl.select(keep)


def cache_user_table(cfg: DictConfig) -> Path:
    out_dir = Path(cfg.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "user_table.parquet"
    build_user_table(cfg).write_parquet(path)
    return path


@dataclass
class PreparedSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    ids: dict[str, np.ndarray]
    fitted: dict = field(default_factory=dict)


def _assign_split(user_id: pl.Series, seed: int, cfg: DictConfig) -> np.ndarray:
    """User-disjoint split via seed-salted hash → 'train'/'val'/'test'."""
    bucket = (user_id.hash(seed=seed) % 100).to_numpy()
    tr = int(cfg.partition.train_frac * 100)
    va = tr + int(cfg.partition.val_frac * 100)
    out = np.where(bucket < tr, "train", np.where(bucket < va, "val", "test"))
    return out


def prepare(cfg: DictConfig, seed: int, table: pl.DataFrame | None = None,
            label: str = "label_purchase") -> PreparedSplit:
    """Load cached user table, user-disjoint split, TRAIN-ONLY standardization."""
    if table is None:
        table = pl.read_parquet(Path(cfg.processed_dir) / "user_table.parquet")

    feat_cols = [c for c in table.columns
                 if c not in ("user_id", "label_purchase", "label_any_event")]
    X = table.select(feat_cols).to_numpy().astype(np.float64)
    y = table[label].cast(pl.Int64).to_numpy()
    uid = table["user_id"].to_numpy()
    split = _assign_split(table["user_id"], seed, cfg)

    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    scaler = StandardScaler().fit(X[tr])
    Xz = scaler.transform(X)

    return PreparedSplit(
        X_train=Xz[tr], X_val=Xz[va], X_test=Xz[te],
        y_train=y[tr], y_val=y[va], y_test=y[te],
        feature_names=feat_cols,
        ids={"train": uid[tr], "val": uid[va], "test": uid[te]},
        fitted={"scaler": scaler},
    )
