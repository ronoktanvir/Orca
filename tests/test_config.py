"""Config: configs/default.yaml loads and exposes the §15 knobs."""

from __future__ import annotations

from config import DEFAULT_CONFIG_PATH, OrcaSettings, load_config


def test_default_config_file_exists():
    assert DEFAULT_CONFIG_PATH.exists()


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, OrcaSettings)
    assert cfg.run.seed == "A"
    assert cfg.run.t_max == 600
    assert cfg.run.day_length == 100
    assert cfg.reward.weights == {"deaths": 0.02, "invalid": 0.05, "idle": 0.05}
    assert cfg.bandit.epsilon == 0.2
    assert cfg.phases.memory_cap == 8
    assert cfg.seeds.train == ["A", "T2", "T3"]
    assert cfg.seeds.heldout == ["B", "C"]


def test_missing_file_returns_defaults(tmp_path):
    cfg = load_config(tmp_path / "does_not_exist.yaml")
    assert isinstance(cfg, OrcaSettings)
    assert cfg.run.seed == "A"


def test_partial_config_merges_defaults(tmp_path):
    p = tmp_path / "partial.yaml"
    p.write_text("run:\n  seed: T2\n  n_episodes: 3\n")
    cfg = load_config(p)
    assert cfg.run.seed == "T2"
    assert cfg.run.n_episodes == 3
    assert cfg.run.t_max == 600  # default preserved
