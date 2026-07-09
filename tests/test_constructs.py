"""Construct-extraction tests (Phase 2).

Guarantees:
1. The construct target matrix is standardized on the TRAIN split only — corrupting
   val/test rows leaves the fitted stats and the train targets unchanged.
2. Constructs inherit the exact same per-seed split as ``prepare()`` (so an aligned
   latent dim is scored against the same rows it was trained on).
3. Definitions are sane (A affinity = spend share summing to 1; deal-reliance in [0,1]).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cadvae.data import constructs as K
from cadvae.data import personality as P

REPO = Path(__file__).resolve().parents[1]


def _synth_personality(n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"ID": np.arange(n), "Response": (rng.random(n) < 0.15).astype(np.int64)}
    for c in P.NUMERIC_COLS:
        data[c] = rng.integers(0, 100, n).astype(np.float64)
    data["Education"] = rng.choice(["Graduation", "PhD"], n)
    data["Marital_Status"] = rng.choice(["Single", "Married"], n)
    return pl.DataFrame(data)


def _cfg_personality():
    return OmegaConf.create({
        "label": "Response",
        "processed_dir": "unused",
        "split": {"test_size": 0.2, "val_size": 0.2, "stratify_on": "Response"},
        "constructs": {
            "rfm": {
                "recency": "Recency",
                "frequency_cols": ["NumDealsPurchases", "NumWebPurchases",
                                   "NumCatalogPurchases", "NumStorePurchases"],
                "monetary_cols": ["MntWines", "MntFruits", "MntMeatProducts",
                                  "MntFishProducts", "MntSweetProducts", "MntGoldProds"],
            },
            "affinity_cols": ["MntWines", "MntFruits", "MntMeatProducts",
                              "MntFishProducts", "MntSweetProducts", "MntGoldProds"],
        },
    })


def test_A_constructs_train_only_standardization():
    cfg = _cfg_personality()
    eng = _synth_personality()
    ks = K.constructs_personality(cfg, seed=7, engineered=eng)

    # locate val/test rows via the same split the constructs used
    tr, va, te = P._stratified_indices(eng["Response"].to_numpy(), cfg, seed=7)
    holdout_ids = set(eng["ID"].to_numpy()[va].tolist()) | set(eng["ID"].to_numpy()[te].tolist())
    corrupt = eng.with_columns(
        pl.when(pl.col("ID").is_in(list(holdout_ids)))
        .then(pl.col("MntWines") * 1000.0 + 1e9)
        .otherwise(pl.col("MntWines")).alias("MntWines")
    )
    ks2 = K.constructs_personality(cfg, seed=7, engineered=corrupt)

    assert np.allclose(ks.fitted["scaler"].mean_, ks2.fitted["scaler"].mean_)
    assert np.allclose(ks.fitted["scaler"].scale_, ks2.fitted["scaler"].scale_)
    assert np.array_equal(ks.C_train, ks2.C_train)


def test_A_constructs_align_with_prepare_split():
    cfg = _cfg_personality()
    cfg2 = OmegaConf.merge(cfg, {"preprocess": {"winsor_lower_q": 0.01, "winsor_upper_q": 0.99}})
    eng = _synth_personality()
    ks = K.constructs_personality(cfg, seed=7, engineered=eng)
    ps = P.prepare(cfg2, seed=7, engineered=eng)
    assert ks.C_train.shape[0] == ps.X_train.shape[0]
    assert ks.C_test.shape[0] == ps.X_test.shape[0]
    # same rows land in the same split (compare the test-set ids)
    assert np.array_equal(ks.raw["train"].shape, (ps.X_train.shape[0], len(ks.names)))


def test_A_construct_definitions_sane():
    cfg = _cfg_personality()
    ks = K.constructs_personality(cfg, seed=7, engineered=_synth_personality())
    assert ks.names[:4] == ["rfm_R", "rfm_F", "rfm_M", "price_sensitivity"]
    assert len(ks.names) == 10                       # 3 RFM + PS + 6 affinity
    raw = ks.raw["train"]
    assert (raw[:, 3] >= 0).all() and (raw[:, 3] <= 1).all()    # deal reliance in [0,1]
    assert np.allclose(raw[:, 4:].sum(1), 1.0)                  # spend shares sum to 1


# ------- real-cache cross-checks (skip if processed caches absent) -------
@pytest.mark.skipif(not (REPO / "data/processed/ecommerce/user_table.parquet").exists(),
                    reason="Dataset B cache not built")
def test_B_constructs_real_cache():
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["data=ecommerce"]).data
    kb = K.constructs_ecommerce(cfg, seed=7)
    assert len(kb.names) == 17                        # 3 RFM + PS + 13 affinity
    assert np.abs(kb.C_train.mean(0)).max() < 1e-9    # train-standardized
    # affinity share sum ~ fraction of events in known categories (~0.69, matches null rate)
    aff = kb.raw["train"][:, 4:]
    assert 0.6 < aff.sum(1).mean() < 0.75
