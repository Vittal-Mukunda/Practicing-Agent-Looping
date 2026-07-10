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
    no label-window information enters the feature vocabulary). Name tiebreak makes
    the vocabulary (→ feature column order) fully deterministic."""
    n = int(cfg.features.top_n_categories)
    vocab = (
        _scan(cfg)
        .filter((pl.col("ts") >= fs) & (pl.col("ts") < ls) & pl.col("top_cat").is_not_null())
        .group_by("top_cat")
        .agg(pl.len().alias("n"))
        .sort(["n", "top_cat"], descending=[True, False])
        .head(n)
        .collect(engine="streaming")
    )
    return vocab["top_cat"].to_list()


def _category_lookup(cfg: DictConfig) -> pl.DataFrame:
    """category_id → top-level category, from all non-null category_code rows (D-016).

    Recovers the 31.84% of events with a null category_code but a valid category_id.
    This is static product-taxonomy metadata, used ONLY to decode the next-category
    LABEL — never as a feature — so no temporal restriction applies. ``min()`` makes
    the (in practice unique) mapping deterministic."""
    return (
        _scan(cfg)
        .filter(pl.col("top_cat").is_not_null())
        .group_by("category_id")
        .agg(pl.col("top_cat").min().alias("top_cat_lookup"))
        .collect(engine="streaming")
    )


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
    _lab_purch = pl.col("in_label") & pl.col("is_purch")
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
        # labels (label window) — D-016
        _lab_purch.any().alias("label_purchase"),
        pl.col("in_label").any().alias("label_any_event"),
        # next-category label: category_id of the FIRST label-window purchase
        # (time order, product_id tiebreak → deterministic); decoded post-agg.
        # Written as ONE sorted column (mask applied elementwise BEFORE the sort):
        # filter() inside sort_by() keys is rejected by the streaming engine, and
        # sorting values and mask separately could mis-align on tied keys.
        pl.when(_lab_purch).then(pl.col("category_id")).otherwise(None)
        .sort_by("ts", "product_id")
        .drop_nulls().first().alias("__label_next_cat_id"),
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

    # --- decode the next-category label via the taxonomy lookup (D-016) ---
    lookup = _category_lookup(cfg)
    tbl = tbl.join(lookup, left_on="__label_next_cat_id", right_on="category_id",
                   how="left").with_columns(
        pl.when(pl.col("label_purchase"))
        .then(pl.coalesce([pl.col("top_cat_lookup"), pl.lit("unknown")]))
        .otherwise(None)                      # non-purchasers: task undefined (masked)
        .alias("label_next_cat"),
        (~pl.col("label_any_event")).alias("label_churned"),   # dormancy proxy (D-016)
    )

    base = [
        "recency_days", "n_events", "n_views", "n_carts", "n_purch",
        "n_sessions", "n_active_days", "total_purch_value", "mean_purch_value",
        "max_purch_value", "mean_view_price", "cart_abandon_rate", "conversion",
    ]
    keep = ["user_id", *base, *share_names,
            "label_purchase", "label_any_event", "label_churned", "label_next_cat"]
    # deterministic row order: streaming group_by emits rows in arbitrary order, and
    # downstream positional operations (train subsampling) must not depend on it (D-028)
    return tbl.select(keep).sort("user_id")


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
    # extra downstream tasks (D-016/D-023): task -> {"train"/"val"/"test": (y, mask), ...}
    labels: dict = field(default_factory=dict)


def _assign_split(user_id: pl.Series, seed: int, cfg: DictConfig) -> np.ndarray:
    """User-disjoint split via seed-salted hash → 'train'/'val'/'test'."""
    bucket = (user_id.hash(seed=seed) % 100).to_numpy()
    tr = int(cfg.partition.train_frac * 100)
    va = tr + int(cfg.partition.val_frac * 100)
    out = np.where(bucket < tr, "train", np.where(bucket < va, "val", "test"))
    return out


def _next_cat_codes(table: pl.DataFrame,
                    feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Encode label_next_cat as int codes over the FEATURE-WINDOW category vocabulary.

    The class space = the cat_share_* feature vocabulary (+ "other" for label-window
    categories outside it, + "unknown" for undecodable ones) — temporally safe: no
    label/test information defines the classes. Mask = label defined (purchasers)."""
    vocab = sorted(c.removeprefix(CATEGORY_PREFIX) for c in feat_cols
                   if c.startswith(CATEGORY_PREFIX)
                   and c not in (f"{CATEGORY_PREFIX}unknown", f"{CATEGORY_PREFIX}other"))
    classes = [*vocab, "other", "unknown"]
    pos = {c: i for i, c in enumerate(classes)}
    raw = table["label_next_cat"].to_numpy()
    mask = np.array([v is not None for v in raw])
    codes = np.array([pos.get(v, pos["other"]) if v is not None else -1 for v in raw],
                     dtype=np.int64)
    return codes, mask, classes


def prepare(cfg: DictConfig, seed: int, table: pl.DataFrame | None = None,
            label: str = "label_purchase") -> PreparedSplit:
    """Load cached user table, user-disjoint split, TRAIN-ONLY standardization."""
    if table is None:
        table = pl.read_parquet(Path(cfg.processed_dir) / "user_table.parquet")
    # deterministic row order regardless of how the cache was written (D-028)
    table = table.sort("user_id")

    # prefix-based exclusion: NOTHING named label_* may enter the features (guarded
    # by a leakage test) — future label columns cannot silently leak
    feat_cols = [c for c in table.columns
                 if c != "user_id" and not c.startswith("label_")]
    X = table.select(feat_cols).to_numpy().astype(np.float64)
    y = table[label].cast(pl.Int64).to_numpy()
    uid = table["user_id"].to_numpy()
    split = _assign_split(table["user_id"], seed, cfg)

    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    scaler = StandardScaler().fit(X[tr])
    Xz = scaler.transform(X)

    # extra downstream tasks (D-016/D-023), aligned to the same split
    labels: dict = {}
    if "label_churned" in table.columns:
        yc = table["label_churned"].cast(pl.Int64).to_numpy()
        all_mask = np.ones(len(yc), dtype=bool)
        labels["churned"] = {"type": "binary",
                             "train": (yc[tr], all_mask[tr]),
                             "val": (yc[va], all_mask[va]),
                             "test": (yc[te], all_mask[te])}
    if "label_next_cat" in table.columns:
        codes, mask, classes = _next_cat_codes(table, feat_cols)
        labels["next_category"] = {"type": "multiclass", "classes": classes,
                                   "train": (codes[tr], mask[tr]),
                                   "val": (codes[va], mask[va]),
                                   "test": (codes[te], mask[te])}

    return PreparedSplit(
        X_train=Xz[tr], X_val=Xz[va], X_test=Xz[te],
        y_train=y[tr], y_val=y[va], y_test=y[te],
        feature_names=feat_cols,
        ids={"train": uid[tr], "val": uid[va], "test": uid[te]},
        fitted={"scaler": scaler},
        labels=labels,
    )
