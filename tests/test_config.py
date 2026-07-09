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


def test_phase1_split_parameters_locked_to_approved_values():
    """Phase 0 held these consequential values at null (D-007 undecided-guard).
    They were decided at Phase 1 (owner-delegated: D-014 stratified 60/20/20 on
    Response; D-015 temporal cut 2019-11-22). This test now LOCKS the approved
    values so they cannot drift silently — strictly stronger than the old null
    guard, not a weakening of it (CLAUDE.md §3). Changing a split value here is a
    deliberate, reviewable edit that must fail this test until updated in lockstep
    with a new D-entry."""
    a = _compose(["data=personality"]).data.split
    assert a.test_size == 0.2 and a.val_size == 0.2 and a.stratify_on == "Response"
    b = _compose(["data=ecommerce"]).data.split
    assert b.feature_window_start == "2019-10-01T00:00:00Z"
    assert b.label_window_start == "2019-11-22T00:00:00Z"
    assert b.label_window_end == "2019-11-30T23:59:59Z"
