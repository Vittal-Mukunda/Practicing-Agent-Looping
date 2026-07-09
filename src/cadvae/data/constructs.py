"""Named marketing constructs — the CA-DVAE alignment-head targets (Phase 2).

Constructs (RFM, price sensitivity, category affinity) are the *named* axes the
model's construct-alignment head learns to predict from a designated subset of
latent dims — this is the interpretability claim (CLAUDE.md core idea #2).

Leakage protocol (CLAUDE.md): constructs are computed from the pre-standardization
tables, split with the **exact same per-seed partition** as ``prepare()``, and the
target matrix is **standardized on the TRAIN split only** (applied to val/test
without refitting). A guard test proves corrupting val/test cannot move the fitted
stats. Raw (unstandardized) values are also returned for persona-card interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from cadvae.data import ecommerce as E
from cadvae.data import personality as P


@dataclass
class ConstructSet:
    """Standardized construct targets (alignment-head labels) for one seed."""

    C_train: np.ndarray
    C_val: np.ndarray
    C_test: np.ndarray
    names: list[str]
    raw: dict[str, np.ndarray]      # split -> unstandardized construct matrix (interpretation)
    fitted: dict = field(default_factory=dict)


def _finalize(raw: np.ndarray, names: list[str], tr, va, te) -> ConstructSet:
    scaler = StandardScaler().fit(raw[tr])
    C = scaler.transform(raw)
    return ConstructSet(
        C_train=C[tr], C_val=C[va], C_test=C[te],
        names=names,
        raw={"train": raw[tr], "val": raw[va], "test": raw[te]},
        fitted={"scaler": scaler},
    )


def constructs_personality(cfg: DictConfig, seed: int,
                           engineered: pl.DataFrame | None = None) -> ConstructSet:
    if engineered is None:
        path = Path(cfg.processed_dir) / "engineered.parquet"
        engineered = pl.read_parquet(path) if path.exists() else P.engineer(cfg)
    c = cfg.constructs

    def _sum(cols):
        return engineered.select(list(cols)).sum_horizontal().to_numpy().astype(np.float64)

    recency = engineered[c.rfm.recency].to_numpy().astype(np.float64)
    freq = _sum(c.rfm.frequency_cols)
    monetary = _sum(c.rfm.monetary_cols)
    deals = engineered["NumDealsPurchases"].to_numpy().astype(np.float64)
    price_sens = deals / np.maximum(freq, 1.0)                      # deal reliance

    aff = engineered.select(list(c.affinity_cols)).to_numpy().astype(np.float64)
    aff_share = aff / np.maximum(monetary[:, None], 1.0)           # spend share per family

    raw = np.column_stack([recency, freq, monetary, price_sens, aff_share])
    def _fam(col):
        return col.replace("Mnt", "").replace("Products", "").replace("Prods", "").lower()
    names = ["rfm_R", "rfm_F", "rfm_M", "price_sensitivity"] + \
            [f"affinity_{_fam(col)}" for col in c.affinity_cols]

    y = engineered[cfg.label].to_numpy()
    tr, va, te = P._stratified_indices(y, cfg, seed)
    return _finalize(raw, names, tr, va, te)


def constructs_ecommerce(cfg: DictConfig, seed: int,
                         table: pl.DataFrame | None = None) -> ConstructSet:
    if table is None:
        table = pl.read_parquet(Path(cfg.processed_dir) / "user_table.parquet")
    c = cfg.constructs

    recency = table[c.rfm.recency].to_numpy().astype(np.float64)
    freq = table[c.rfm.frequency].to_numpy().astype(np.float64)
    monetary = table[c.rfm.monetary].to_numpy().astype(np.float64)
    # price sensitivity = price-tier proxy; negate so higher = cheaper browsing = more sensitive
    price_sens = -table["mean_view_price"].to_numpy().astype(np.float64)

    excl = set(c.affinity_exclude)
    aff_cols = [col for col in table.columns
                if col.startswith(c.affinity_prefix) and col not in excl]
    aff = table.select(aff_cols).to_numpy().astype(np.float64)

    raw = np.column_stack([recency, freq, monetary, price_sens, aff])
    names = ["rfm_R", "rfm_F", "rfm_M", "price_sensitivity"] + \
            [col.replace("cat_share_", "affinity_") for col in aff_cols]

    split = E._assign_split(table["user_id"], seed, cfg)
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    return _finalize(raw, names, tr, va, te)
