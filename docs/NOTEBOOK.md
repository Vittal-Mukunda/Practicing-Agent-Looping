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

---

## 2026-07-09 — Session 3 (Phase 0 data acquisition + schema inspection)

Owner provided a Kaggle API token (`KGAT_…`) → unblocked 0.10/0.11.

**Auth + downloads (all commands + gotchas in data/README.md):**
- Token → `KAGGLE_API_TOKEN` env var, persisted to `~/.kaggle/access_token`
  (outside repo, chmod 600). kaggle run via `uv tool run --from kaggle` (isolated;
  NOT a project dep — `uv run kaggle` tried to rebuild the editable pkg and hit a
  Windows "Access is denied" lock on `.venv/.../cadvae-0.1.0.dist-info`). For all
  Polars inspection I called `.venv/Scripts/python.exe` directly to avoid that lock.
- **Dataset A**: `marketing_campaign.csv`, 220,188 B, sha256
  `cd0affa36b1b981e80ba0e27767e9b3ab723f7a3dba948f722af81abc6b990ea`.
  License printed by CLI: **CC0-1.0** (resolves A-half of D-011).
- **Dataset B**: `2019-Oct.csv` only (5,668,612,855 B). `-f` single-file download
  ignores `--unzip`; extracted with `unzip`. License printed: **`copyright-authors`**
  (restrictive — contradicts HF-mirror wording; D-011 now needs owner review).
  Nov deferred to Phase 1 (same schema; 9 GB, no gate value now).

**Schema inspection (real, not assumed — scripts in session scratchpad):**
- A (`read_csv`, tab-sep): 2240×29 confirmed. `Response` pos-rate 0.149. `Income`
  24 nulls + 666,666 outlier (p99 94,472). `Year_Birth` 3 rows <1920. `Marital_Status`
  junk levels Absurd/Alone/YOLO (7 rows). `Z_CostContact`/`Z_Revenue` constant.
- B-Oct (`scan_csv` + `engine="streaming"`, 5.67 GB in ~59 s, bounded RAM): 9 cols,
  **42,448,764 rows**. Nulls: category_code 31.84%, brand 14.40%, user_session 2.
  event_type = view 96.07% / cart 2.18% / **purchase 1.75%** (→ PR-AUC primary).
  **No `remove_from_cart` in Oct** (docs claimed it — do not hardcode). 3,022,290
  users; category_id 0-null (reliable key) vs category_code 31.84% null. price
  0–2574.07, 0 negatives, 68,673 zeros. Time span full Oct in UTC.

**Both data cards COMPLETED** (docs/data_cards/). D-009 resolved (repo at C:\VAE,
734 GB free, not OneDrive). Data files confirmed gitignored (`data/*`, only
README tracked). No src/ code changed → no test re-run needed; scaffold suite
still green from Session 2.

**Phase 0 now at the GATE.** Awaiting owner sign-off (0.13). Open items for owner:
D-011-B license call, D-004 git identity check, D-005 (results/ gitignored — assent),
D-008 (GBT head — can defer to Phase 3). Nothing else blocks. Next phase after
sign-off = Phase 1 (Polars lazy preprocessing + leakage-safe splits + Nov download).
