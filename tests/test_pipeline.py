"""Phase 1 integration test — the data pipeline runs end-to-end and emits the
expected processed artifacts with a valid frozen-representation contract.

(The *full* frozen-representation run with a downstream head lands in Phase 3,
once baselines exist; here we assert the data contract those runs depend on.)
Skips cleanly when the processed caches are absent (fresh clone before caching).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from cadvae.data import ecommerce as E
from cadvae.data import personality as P

REPO = Path(__file__).resolve().parents[1]


def _cfg(dataset: str):
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        return compose(config_name="config", overrides=[f"data={dataset}"]).data


def _assert_contract(s, expected_total: int | None = None):
    # feature-matrix width matches the declared feature names
    assert s.X_train.shape[1] == len(s.feature_names)
    assert s.X_val.shape[1] == len(s.feature_names)
    assert s.X_test.shape[1] == len(s.feature_names)
    # labels are binary and aligned with rows
    for X, y in [(s.X_train, s.y_train), (s.X_val, s.y_val), (s.X_test, s.y_test)]:
        assert X.shape[0] == y.shape[0]
        assert set(np.unique(y)).issubset({0, 1})
        assert not np.isnan(X).any()
    if expected_total is not None:
        assert len(s.y_train) + len(s.y_val) + len(s.y_test) == expected_total


@pytest.mark.skipif(not (REPO / "data/processed/personality/engineered.parquet").exists(),
                    reason="Dataset A engineered cache not built")
def test_personality_pipeline_contract():
    s = P.prepare(_cfg("personality"), seed=7)
    _assert_contract(s, expected_total=2240)
    assert len(s.feature_names) == 34  # 23 numeric + Education(5) + Marital(6)


@pytest.mark.skipif(not (REPO / "data/processed/ecommerce/user_table.parquet").exists(),
                    reason="Dataset B user_table cache not built")
def test_ecommerce_pipeline_contract():
    s = E.prepare(_cfg("ecommerce"), seed=7)
    _assert_contract(s)
    assert len(s.feature_names) == 27          # 13 behavioral + 13 known cats + unknown
    assert 0.02 < s.y_train.mean() < 0.05      # ~3.3% positive (sane range)
