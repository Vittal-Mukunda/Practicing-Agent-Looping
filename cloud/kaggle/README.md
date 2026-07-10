# Kaggle Setup — VERIFIED path (use `kaggle_cells.py`)

Free P100 GPU (16 GB VRAM), 30 h/week quota. This flow was verified against the
actual repo code (config paths, cache requirements, split determinism).

**Key design decisions (why this differs from a naive setup):**

- **No raw data on Kaggle.** The sweep only reads the two processed parquet
  caches (`engineered.parquet`, `user_table.parquet`, ~68 MB total). You upload
  your local `data/processed` as a **private** Kaggle dataset. This skips the
  14 GB raw download + cache rebuild AND guarantees the caches are bit-identical
  to the ones the local Phase-3 baseline bars were computed on.
  (Private is mandatory: dataset B's license is `copyright-authors`.)
- **polars pinned to 1.42.1, scikit-learn to 1.9.0** (exact local versions).
  Dataset B's train/val/test split is a seed-salted *polars hash* of user_id,
  and polars hashes are NOT stable across versions — an unpinned polars would
  silently produce different splits than the local baselines. sklearn is pinned
  so the frozen-rep heads are identical to the Phase-3 protocol.
- **Results persist three ways:** (1) run via *Save & Run All* — a completed
  version's output is stored permanently by Kaggle; (2) a daemon thread pushes
  `results/phase5` to the `kaggle-results` GitHub branch every 15 min (survives
  session timeout/crash); (3) each new session restores from that branch first,
  so the sweep's per-run resume-skip continues across rounds.

## Setup (~30 min, one-time)

### 1. GitHub token (5 min)

github.com → Settings → Developer settings → Personal access tokens →
**Fine-grained tokens** → Generate new token:
- Repository access: **Only** `Practicing-Agent-Looping`
- Permissions → Contents: **Read and write**

Copy the token.

### 2. Upload processed caches as a private Kaggle dataset (10 min)

kaggle.com → Datasets → **New Dataset** → upload these two local files:
- `C:\VAE\data\processed\personality\engineered.parquet`
- `C:\VAE\data\processed\ecommerce\user_table.parquet`

Name it `cadvae-processed`, visibility **Private**, Create.

### 3. Create the notebook (15 min)

kaggle.com/code → **New Notebook**, then:
- **Settings** (right sidebar): Accelerator = **GPU P100**, Internet = **On**
  (requires a phone-verified Kaggle account)
- **Add-ons → Secrets**: add secret `GITHUB_TOKEN` = the token from step 1,
  and make sure it is *attached* to this notebook
- **Add Input**: attach your private `cadvae-processed` dataset
  (Datasets → Your Work)
- Paste the 6 cells from `kaggle_cells.py` (each `%% CELL` block = one cell)

### 4. Smoke-test, then launch

1. Run cells 1–3 interactively (a few minutes, no GPU time wasted). They verify:
   clone, pins, GPU, caches, GitHub push. If all three pass, stop the session.
2. **Save Version → Save & Run All (Commit).** Close the browser — it runs in
   the background on Kaggle's side.

## Monitoring & recovery

- Watch progress: the version's **Logs** on Kaggle, or the commit stream on the
  `kaggle-results` branch (a commit every ≤15 min while it's running).
- **If the version times out or fails** (dataset B may exceed the batch session
  limit): just **Save & Run All again**. The new session restores completed
  runs from GitHub and continues — nothing is recomputed. Repeat until cell 6
  prints `ALL 288 RUNS COMPLETE`.

## Bring results home

```powershell
New-Item -ItemType Directory -Force C:\VAE\results\phase5 | Out-Null
git clone --branch kaggle-results --single-branch `
    https://github.com/Vittal-Mukunda/Practicing-Agent-Looping.git $env:TEMP\cadvae-results
Copy-Item -Recurse -Force $env:TEMP\cadvae-results\phase5\* C:\VAE\results\phase5\
```

Then run Phase 6 locally (needs the local parquet caches + Phase-3 records,
both already on your machine):

```powershell
.venv\Scripts\python.exe -m cadvae.eval.run_phase6
```

## Honest caveats

- Kaggle's GPU batch session limit (~9–12 h) means dataset B will likely need
  **two Save & Run All rounds**. The restore/resume design makes this safe.
- Kaggle's per-core CPU is slower than the local Ryzen 9; the single-threaded
  (D-028) head fits dominate B's per-run time, so expect ~220–300 s/run vs the
  local 179.5 s anchor.
- Numbers will not be bit-identical to a local 4050 run (different GPU —
  documented unavoidable nondeterminism). Splits, heads, and protocol are
  identical thanks to the pins and the shared caches.
