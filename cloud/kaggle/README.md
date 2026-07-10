# Kaggle Setup (Recommended)

Free P100 GPU (16 GB VRAM), 30 hours/week. Datasets already on Kaggle.

## Steps (20 minutes)

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. **Settings** (right sidebar):
   - Accelerator: **GPU P100**
   - Internet: **On** (needed for git clone + pip install)
   - Persistence: **Files only** (so results survive session restart)
3. **Add Data** (right sidebar) — search and add:
   - `iakash17/customer-personality-analysis`
   - `mkechinov/ecommerce-behavior-data-from-multi-category-store`
4. Copy-paste the cells from `notebook.py` into the Kaggle notebook
5. Run all cells

## Important notes

- The **30 hours/week** GPU quota is enough for both sweeps (~8h total)
- Both sweep commands are **resumable** — if the session dies, just relaunch
- Results appear in `/kaggle/working/cadvae/results/phase5/`
- Download results from the notebook's **Output** tab when done
- Copy results back to your local `results/phase5/` for Phase 6 analysis

## After the sweep

Copy the `results/phase5/` folder back to your local machine, then run Phase 6
locally (it's CPU-only analysis, takes seconds):

```bash
.venv\Scripts\python.exe -m cadvae.eval.run_phase6
```
