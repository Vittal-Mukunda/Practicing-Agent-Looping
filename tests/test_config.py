"""Hydra config schema tests: every group entry must compose, and the
split-type invariants mandated by CLAUDE.md are locked here so they cannot
drift silently (stratified for Dataset A, temporal for Dataset B)."""

from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=overrides or [])


def test_default_config_composes_and_resolves():
    cfg = _compose()
    assert cfg.seed == 7
    assert cfg.device == "cuda"
    assert cfg.model.name == "cadvae"
    assert cfg.data.name == "personality"
    assert cfg.run_name == "cadvae_personality_seed7"  # interpolation resolves


def test_every_data_group_composes():
    for name in ["personality", "ecommerce"]:
        assert _compose([f"data={name}"]).data.name == name


def test_every_model_group_composes():
    for name in ["cadvae", "ae"]:
        assert _compose([f"model={name}"]).model.name == name


def test_split_types_are_locked_per_claude_md():
    assert _compose(["data=personality"]).data.split.type == "stratified"
    assert _compose(["data=ecommerce"]).data.split.type == "temporal"


def test_undecided_phase1_parameters_are_null_not_defaults():
    """Split boundaries/ratios are consequential decisions awaiting the Phase 1
    gate; they must be explicitly null so nothing can run on unapproved values."""
    a = _compose(["data=personality"]).data.split
    assert a.test_size is None and a.val_size is None and a.stratify_on is None
    b = _compose(["data=ecommerce"]).data.split
    assert b.train_end is None and b.val_end is None and b.test_end is None
