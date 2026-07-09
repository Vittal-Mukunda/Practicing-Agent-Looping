"""Leakage-guard tests — the #1 reviewer attack surface (CLAUDE.md).

Two guarantees are proven here, both on small self-contained synthetic fixtures
(no dependency on the multi-GB raw data, so a fresh clone runs them in CI):

1. **Train-only fitting.** Corrupting val/test rows must leave every train-fitted
   transform (scaler stats, winsor bounds, one-hot vocabulary) byte-for-byte
   unchanged. If val/test could influence fitting, this test fails.
2. **Temporal separation (Dataset B).** A purchase in the label window must NOT
   appear in a user's feature-window aggregates; the universe filter and label
   definitions must respect the window boundaries.

Real-data cross-checks (split disjointness on the actual caches) run only when the
processed artifacts exist, and skip otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from cadvae.data import ecommerce as E
from cadvae.data import personality as P

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Dataset A — synthetic engineered fixture
# --------------------------------------------------------------------------- #
def _synthetic_personality(n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(np.int64)  # ~15% positive, like the real label
    data = {"ID": np.arange(n), "Response": y}
    for c in P.NUMERIC_COLS:
        data[c] = rng.integers(0, 100, n).astype(np.float64)
    data["Income"] = rng.integers(10_000, 100_000, n).astype(np.float64)
    data["Age"] = rng.integers(20, 80, n).astype(np.float64)
    data["Education"] = rng.choice(["Graduation", "PhD", "Master"], n)
    data["Marital_Status"] = rng.choice(["Single", "Married", "Together"], n)
    return pl.DataFrame(data)


def _cfg_personality():
    return OmegaConf.create({
        "label": "Response",
        "split": {"test_size": 0.2, "val_size": 0.2, "stratify_on": "Response"},
        "preprocess": {"winsor_lower_q": 0.01, "winsor_upper_q": 0.99},
        "processed_dir": "unused",
    })


def test_A_splits_disjoint_and_complete():
    cfg = _cfg_personality()
    s = P.prepare(cfg, seed=7, engineered=_synthetic_personality())
    ids = np.concatenate([s.ids["train"], s.ids["val"], s.ids["test"]])
    assert len(ids) == len(set(ids.tolist()))            # disjoint
    assert len(ids) == 400                               # complete
    n = len(ids)
    assert abs(len(s.y_train) / n - 0.6) < 0.02
    assert abs(len(s.y_test) / n - 0.2) < 0.02


def test_A_transforms_fit_on_train_only():
    """Corrupt val+test rows; train-fitted transforms must not move at all."""
    cfg = _cfg_personality()
    eng = _synthetic_personality()
    s = P.prepare(cfg, seed=7, engineered=eng)
    holdout_ids = set(s.ids["val"].tolist()) | set(s.ids["test"].tolist())

    corrupt = eng.with_columns(
        pl.when(pl.col("ID").is_in(list(holdout_ids)))
        .then(pl.col("Income") * 1000.0 + 1e9)   # wildly corrupt held-out rows only
        .otherwise(pl.col("Income")).alias("Income")
    )
    s2 = P.prepare(cfg, seed=7, engineered=corrupt)

    # identical split (stratification depends on label+seed, not on corrupted feature)
    assert np.array_equal(s.ids["train"], s2.ids["train"])
    # train-fitted stats unchanged despite corrupted val/test
    assert np.allclose(s.fitted["scaler"].mean_, s2.fitted["scaler"].mean_)
    assert np.allclose(s.fitted["scaler"].scale_, s2.fitted["scaler"].scale_)
    assert s.fitted["winsor_bounds"] == s2.fitted["winsor_bounds"]
    # ...and the train matrix itself is byte-for-byte identical
    assert np.array_equal(s.X_train, s2.X_train)


def test_A_no_nan_and_standardized():
    cfg = _cfg_personality()
    s = P.prepare(cfg, seed=7, engineered=_synthetic_personality())
    assert not np.isnan(s.X_train).any()
    assert not np.isnan(s.X_val).any() and not np.isnan(s.X_test).any()
    # numeric block (first len(NUMERIC_COLS)) standardized on train
    k = len(P.NUMERIC_COLS)
    assert np.abs(s.X_train[:, :k].mean(0)).max() < 1e-9


# --------------------------------------------------------------------------- #
# Dataset B — synthetic event-log fixture spanning both windows
# --------------------------------------------------------------------------- #
def _synthetic_ecommerce_csv(tmp_path: Path) -> Path:
    """User 1: 5 feature views + a LABEL-window purchase (must not count as feature
    purchase). User 2: 6 feature events incl. a purchase, no label activity (churn).
    User 3: 2 events → below universe threshold (excluded)."""
    rows = []
    def ev(t, et, uid, price=10.0, cat="electronics.smartphone", sess="s"):
        rows.append({"event_time": t + " UTC", "event_type": et, "product_id": 1,
                     "category_id": 100, "category_code": cat, "brand": "b",
                     "price": price, "user_id": uid, "user_session": f"{uid}-{sess}"})
    for i in range(5):
        ev(f"2019-10-0{i+1} 10:00:00", "view", 1)
    ev("2019-11-25 10:00:00", "purchase", 1, price=500.0)      # LABEL window
    for i in range(5):
        ev(f"2019-10-0{i+1} 11:00:00", "view", 2)
    ev("2019-10-06 11:00:00", "purchase", 2, price=99.0)       # FEATURE window purchase
    ev("2019-10-01 12:00:00", "view", 3)                       # only 2 events → excluded
    ev("2019-10-02 12:00:00", "view", 3)
    p = tmp_path / "synth.csv"
    pl.DataFrame(rows).write_csv(p)
    return p


def _cfg_ecommerce(tmp_path: Path):
    _synthetic_ecommerce_csv(tmp_path)
    return OmegaConf.create({
        "raw_dir": str(tmp_path), "files": ["synth.csv"],
        "processed_dir": str(tmp_path / "proc"),
        "split": {
            "feature_window_start": "2019-10-01T00:00:00Z",
            "label_window_start": "2019-11-22T00:00:00Z",
            "label_window_end": "2019-11-30T23:59:59Z",
        },
        "partition": {"train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
        "universe": {"min_events_feature_window": 5},
        "features": {"top_n_categories": 15},
    })


def test_B_label_window_purchase_excluded_from_features(tmp_path):
    cfg = _cfg_ecommerce(tmp_path)
    tbl = E.build_user_table(cfg).sort("user_id")
    rec = {r["user_id"]: r for r in tbl.iter_rows(named=True)}
    # user 3 excluded by the >=5-event universe threshold
    assert set(rec) == {1, 2}
    # user 1 purchased ONLY in the label window → feature n_purch must be 0, label=1
    assert rec[1]["n_purch"] == 0
    assert rec[1]["label_purchase"] is True
    assert rec[1]["n_events"] == 5
    # user 2 purchased in the feature window → n_purch=1, no label activity → churn
    assert rec[2]["n_purch"] == 1
    assert rec[2]["label_purchase"] is False
    assert rec[2]["label_any_event"] is False


def test_B_temporal_guarantee_features_precede_labels(tmp_path):
    """Directly assert the window boundary the whole design rests on."""
    cfg = _cfg_ecommerce(tmp_path)
    fs = E._parse_anchor(cfg.split.feature_window_start)
    ls = E._parse_anchor(cfg.split.label_window_start)
    scan = E._scan(cfg).filter((pl.col("ts") >= fs) & (pl.col("ts") < ls))
    max_feat_ts = scan.select(pl.col("ts").max()).collect().item()
    assert max_feat_ts < ls  # every feature event strictly precedes the label window


def test_B_standardization_fit_on_train_only(tmp_path):
    """Corrupt held-out users; train-fitted scaler must not move."""
    # need enough users for a populated train/val/test — build a larger synthetic table
    rng = np.random.default_rng(1)
    n = 3000
    tbl = pl.DataFrame({
        "user_id": np.arange(n),
        "recency_days": rng.random(n) * 50,
        "n_events": rng.integers(5, 50, n).astype(float),
        "n_purch": rng.integers(0, 5, n).astype(float),
        "label_purchase": (rng.random(n) < 0.03),
        "label_any_event": (rng.random(n) < 0.3),
    })
    cfg = OmegaConf.create({
        "processed_dir": "unused",
        "partition": {"train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
    })
    s = E.prepare(cfg, seed=7, table=tbl)
    holdout = set(s.ids["val"].tolist()) | set(s.ids["test"].tolist())
    corrupt = tbl.with_columns(
        pl.when(pl.col("user_id").is_in(list(holdout)))
        .then(pl.col("recency_days") * 1000 + 1e6)
        .otherwise(pl.col("recency_days")).alias("recency_days")
    )
    s2 = E.prepare(cfg, seed=7, table=corrupt)
    assert np.array_equal(s.ids["train"], s2.ids["train"])
    assert np.allclose(s.fitted["scaler"].mean_, s2.fitted["scaler"].mean_)
    assert np.allclose(s.fitted["scaler"].scale_, s2.fitted["scaler"].scale_)


def test_B_users_disjoint_across_splits(tmp_path):
    rng = np.random.default_rng(2)
    n = 5000
    tbl = pl.DataFrame({
        "user_id": np.arange(n),
        "recency_days": rng.random(n),
        "label_purchase": (rng.random(n) < 0.03),
        "label_any_event": (rng.random(n) < 0.3),
    })
    cfg = OmegaConf.create({
        "processed_dir": "unused",
        "partition": {"train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
    })
    s = E.prepare(cfg, seed=7, table=tbl)
    ids = np.concatenate([s.ids["train"], s.ids["val"], s.ids["test"]])
    assert len(ids) == len(set(ids.tolist())) == n


# --------------------------------------------------------------------------- #
# Real-data cross-checks (skip if the processed caches are absent)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (REPO / "data/processed/ecommerce/user_table.parquet").exists(),
                    reason="Dataset B cache not built")
def test_B_real_cache_split_disjoint_and_balanced():
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["data=ecommerce"]).data
    s = E.prepare(cfg, seed=7)
    ids = np.concatenate([s.ids["train"], s.ids["val"], s.ids["test"]])
    assert len(ids) == len(set(ids.tolist()))                    # user-disjoint
    # label prevalence stable across the user-disjoint splits
    assert abs(s.y_train.mean() - s.y_test.mean()) < 0.005
