"""Run-logging tests: config hash stability/sensitivity and run-dir contents."""

import json

from omegaconf import OmegaConf

from cadvae.utils.run_logging import RunLogger, config_hash


def _cfg(seed: int = 7, beta: float = 1.0):
    return OmegaConf.create({"seed": seed, "model": {"name": "x", "beta": beta}})


def test_config_hash_stable_for_identical_configs():
    assert config_hash(_cfg()) == config_hash(_cfg())


def test_config_hash_changes_when_any_value_changes():
    assert config_hash(_cfg()) != config_hash(_cfg(seed=8))
    assert config_hash(_cfg()) != config_hash(_cfg(beta=4.0))


def test_config_hash_ignores_key_order():
    a = OmegaConf.create({"x": 1, "y": 2})
    b = OmegaConf.create({"y": 2, "x": 1})
    assert config_hash(a) == config_hash(b)


def test_run_dir_contains_config_meta_and_metrics(tmp_path):
    cfg = _cfg()
    logger = RunLogger(tmp_path / "run1", cfg)
    logger.log_metrics({"loss": 1.5}, step=0)
    logger.log_metrics({"loss": 1.2}, step=1)

    assert (tmp_path / "run1" / "config.yaml").exists()
    meta = json.loads((tmp_path / "run1" / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 7
    assert meta["config_hash"] == config_hash(cfg)
    assert "started_utc" in meta

    lines = (tmp_path / "run1" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["loss"] for line in lines] == [1.5, 1.2]
    assert json.loads(lines[0])["step"] == 0
