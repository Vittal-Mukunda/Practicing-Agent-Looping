# NOTEBOOK.md — experiment log

Every experiment records: config hash, seed, command, result, observation.
No experiments have been run yet (Phase 0 = setup only, by design).

---

## 2026-07-09 — Session 1 (Phase 0)

**Done:**
- Repo bootstrapped from empty directory. Baseline commit `64415ec` (CLAUDE.md only).
- Environment: Python 3.12.10, uv 0.11.28, `uv.lock` resolved (48 packages,
  win32-only), torch 2.6.0+cu124 from the PyTorch CUDA index.
- Scaffold: Hydra config skeleton, `cadvae` package (utils implemented:
  seeding / device / run_logging; data/models/eval/viz are placeholders by design),
  pytest harness (4 test modules), docs (DECISIONS, CHECKLIST, data-card skeletons).
- `uv sync --all-groups` completed (exit 0): .venv created, torch 2.6.0+cu124
  installed. Lockfile verified to pin the CUDA wheel (cp312, win_amd64, sha256),
  not a CPU wheel.

**Session paused by owner BEFORE verification ran. Nothing is verified yet:
no CUDA check, no pytest, no lint/type-check, scaffold not committed. Do not
mark 0.4/0.8/0.9 complete without running them. First actions on resume:**
```powershell
python -m uv run python -m cadvae.utils.device   # must show RTX 4050, cuda_available true
python -m uv run pytest                          # all 4 test modules must pass
python -m uv run ruff check src tests
python -m uv run mypy
# then: git add -A; git commit (Phase 0 scaffold); paste evidence here; gate summary
```

**Blocked / pending (owner):**
- Datasets not on disk; no Kaggle credentials on machine → schema inspection and
  final data cards blocked (checklist 0.10/0.11). Download instructions: data/README.md.
- License confirmation on both Kaggle pages (D-011).
- Open decisions at gate: D-005 (results/ gitignored), D-008 (GBT head choice),
  D-009 (OneDrive location — recommend relocating before the 15 GB download).

**Observations:**
- `nvidia-smi` requires admin on this machine (fails with a permissions error even
  unsandboxed) — GPU facts are gathered via torch/CUDA runtime instead, which is
  what matters for training anyway.
- pip's first attempt to install uv hit a network read timeout; succeeded on retry
  with `--default-timeout=120`. Worth remembering for the big dataset downloads.

---

## 2026-07-09 — Session 2 (Phase 0 verification)

**Verification evidence (all four resume commands run):**
- `python -m uv run python -m cadvae.utils.device` → torch 2.6.0+cu124,
  cuda_available=true, NVIDIA GeForce RTX 4050 Laptop GPU, 6.0 GB VRAM,
  cuda runtime 12.4, cudnn 90100, compute capability 8.9, gpu_matmul_ok=true
  (9.1 MB allocated after matmul). No silent CPU fallback.
- `pytest` → **15 passed** (7.03 s).
- `ruff check src tests` → 1 finding (UP017, `timezone.utc` → `datetime.UTC` in
  `run_logging.py`); `--fix` applied (3 mechanical fixes incl. import cleanup),
  re-check clean. pytest re-run after fix: 15 passed.
- `mypy` → Success: no issues found in 9 source files.

**Result:** checklist 0.4 / 0.8 / 0.9 flipped to Completed. Scaffold committed
(this commit). Phase 0 remaining items are all owner-blocked: 0.10 datasets
(Kaggle creds), 0.11 data cards (depends on 0.10), 0.12 licenses (D-011),
0.13 gate sign-off (D-005, D-008, D-009 OneDrive, D-004 git identity).
