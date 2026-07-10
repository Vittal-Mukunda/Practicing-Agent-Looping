# Google Colab Setup

Free T4 GPU (15 GB VRAM). Familiar Jupyter interface.

## Steps (15 minutes)

1. Upload `CA_DVAE_Sweep.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. Run cells in order — you'll be prompted to upload your `kaggle.json` API token
   (get it from kaggle.com → Account → Create New Token)
4. The sweeps run sequentially (~10h total)

## Important notes

- Colab free tier has a **12-hour session limit** — the A sweep (~2h) will finish
  easily; B (~8h) might get cut. If it does, reconnect and relaunch — the sweep
  is resumable.
- Colab Pro ($10/mo) gives longer sessions and V100/A100 access.
- Results are in `/content/cadvae/results/phase5/`. The last cell zips and
  downloads them.

## After the sweep

Copy the `results/phase5/` folder back to your local machine, then run Phase 6
locally:

```bash
.venv\Scripts\python.exe -m cadvae.eval.run_phase6
```
