# Cloud Compute Setup

This folder contains self-contained setup files for running the CA-DVAE sweep
on cloud platforms. **The main project runs locally on the RTX 4050 as designed.**
These are optional alternatives if you prefer cloud compute.

| Platform | Cost | GPU | Setup time | Best for |
|---|---|---|---|---|
| **Kaggle** (recommended) | Free | P100 16GB | 20 min | Data already on Kaggle |
| **Google Colab** | Free | T4 15GB | 15 min | Familiar Jupyter interface |
| **HuggingFace Spaces** | ~$6 | T4 16GB | 2-3h | Only if you need a persistent endpoint |

## Quick start

Pick ONE platform. Each subfolder has its own README with exact steps.

```
cloud/
  kaggle/        # VERIFIED against the repo code — use this one
  colab/         # UNVERIFIED sketch (known issues: wrong raw paths, no cache build)
  huggingface/   # UNVERIFIED sketch (same issues + missing 2019-Nov.csv download)
```

**Only the Kaggle path has been verified against the actual configs and code**
(processed-cache requirement, raw path layout, polars-hash split determinism).
The colab/ and huggingface/ folders are early sketches kept for reference; fix
them against `kaggle/kaggle_cells.py` before using.

## What gets run

The sweep command is identical across platforms:
```bash
python -m cadvae.eval.run_cadvae_sweep data=personality
python -m cadvae.eval.run_cadvae_sweep data=ecommerce eval.train_subsample=200000 model.max_epochs=40
```

Both commands are **resumable** — if the session dies, relaunch and it skips completed runs.
