"""Phase 7 — reproduction verification.

Re-runs a pinned, small slice of the protocol (one seed of the Phase-3 baselines
by default) and compares every numeric metric against the RECORDED evidence in
``results/phase3/{data}_baselines/records.json``. The audited pipeline is
bit-reproducible on this machine (D-028), so the default tolerance is tight;
``repro.atol`` loosens it for cross-machine checks (different BLAS/GPU).

Exit code 0 = all compared values match within tolerance; 1 = any mismatch
(printed). This is the "fresh clone reproduces the numbers" gate of the
Definition of Done.

    python -m cadvae.eval.repro_check data=personality
    python -m cadvae.eval.repro_check data=ecommerce \
        eval.train_subsample=200000 model.max_epochs=40
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
from omegaconf import DictConfig

from cadvae.eval.protocol import evaluate_baselines
from cadvae.eval.run_baselines import _prepare, _tasks

_ID_KEYS = ("method", "head", "seed", "task")


def _key(r: dict) -> tuple:
    return tuple(r.get(k, "primary" if k == "task" else None) for k in _ID_KEYS)


def compare_records(new: list[dict], ref: list[dict], atol: float = 1e-9,
                    rtol: float = 0.0) -> list[str]:
    """Match records by (method, head, seed, task); compare numeric fields.
    Returns human-readable mismatch messages (empty list = reproduction OK)."""
    ref_by_key = {_key(r): r for r in ref}
    problems: list[str] = []
    for r in new:
        k = _key(r)
        if k not in ref_by_key:
            problems.append(f"{k}: not present in reference records")
            continue
        rr = ref_by_key[k]
        for m, v in r.items():
            if m in _ID_KEYS or not isinstance(v, (int, float)):
                continue
            if m not in rr:
                problems.append(f"{k}: metric '{m}' missing from reference")
                continue
            w = float(rr[m])
            both_nan = isinstance(v, float) and math.isnan(v) and math.isnan(w)
            if not both_nan and not math.isclose(float(v), w, rel_tol=rtol, abs_tol=atol):
                problems.append(f"{k}: {m} = {v!r} vs reference {w!r}")
    return problems


def run_check(cfg: DictConfig) -> list[str]:
    ref_path = Path(cfg.results_dir) / "phase3" / f"{cfg.data.name}_baselines" / "records.json"
    if not ref_path.exists():
        raise FileNotFoundError(f"{ref_path} not found — no recorded evidence to check "
                                f"against (run Phase 3 first)")
    ref = json.loads(ref_path.read_text())
    seed = int(cfg.repro.seed)
    ps, ks = _prepare(cfg, seed)
    new = evaluate_baselines(ps, ks, cfg, seed, device=cfg.device, tasks=_tasks(ps, cfg))
    return compare_records(new, ref, atol=float(cfg.repro.atol), rtol=float(cfg.repro.rtol))


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    problems = run_check(cfg)
    if problems:
        print(f"REPRODUCTION FAILED ({len(problems)} mismatches):", flush=True)
        for p in problems[:50]:
            print(f"  {p}", flush=True)
        raise SystemExit(1)
    print(f"REPRODUCTION OK: {cfg.data.name} seed {cfg.repro.seed} matches the "
          f"recorded evidence (atol={cfg.repro.atol})", flush=True)


if __name__ == "__main__":
    main()
