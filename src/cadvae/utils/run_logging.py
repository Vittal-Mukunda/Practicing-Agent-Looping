"""Local-first run logging.

CLAUDE.md requirement: every run writes its resolved config, a config hash,
the git commit, the seed, and full metrics under ``results/``. No external
logging service is required (W&B stays off by default).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def config_hash(cfg: DictConfig) -> str:
    """Stable short hash of the fully-resolved config.

    Hashes the JSON serialization of the resolved container with sorted keys,
    so formatting and key order cannot change the hash — only actual values can.
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    canonical = json.dumps(container, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def git_commit() -> str | None:
    """Current commit hash, or None outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes (reproducibility red flag)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class RunLogger:
    """Writes one run directory: resolved config, run metadata, JSONL metrics.

    Layout::

        <run_dir>/config.yaml    resolved config (interpolations materialized)
        <run_dir>/meta.json      config_hash, git_commit, git_dirty, seed, started_utc
        <run_dir>/metrics.jsonl  one JSON record per log_metrics() call
    """

    def __init__(self, run_dir: str | Path, cfg: DictConfig) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        OmegaConf.save(cfg, self.run_dir / "config.yaml", resolve=True)
        meta = {
            "config_hash": config_hash(cfg),
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "seed": cfg.get("seed"),
            "started_utc": datetime.now(UTC).isoformat(),
        }
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._metrics_path = self.run_dir / "metrics.jsonl"

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        record = {"step": step, **metrics}
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
