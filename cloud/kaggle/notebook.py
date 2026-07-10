# %% [markdown]
# # CA-DVAE Sweep — Kaggle Notebook
#
# **Setup instructions:**
# 1. Create a new Kaggle notebook with **GPU P100** accelerator
# 2. Add these two datasets (right sidebar → "Add data"):
#    - `iakash17/customer-personality-analysis`
#    - `mkechinov/ecommerce-behavior-data-from-multi-category-store`
# 3. Paste this entire file into the notebook and run all cells
#
# **Runtime:** ~6-8 hours total (A: ~1.5h, B: ~5-6h). Both sweeps are
# resumable — if the session restarts, relaunch and completed runs are skipped.

# %% Clone repo and install
# fmt: off
"""
!git clone https://github.com/Vittal-Mukunda/Practicing-Agent-Looping.git /kaggle/working/cadvae
%cd /kaggle/working/cadvae
!pip install -e . --quiet
"""
# fmt: on

# %% Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% Link Kaggle datasets to the expected data/raw/ paths
import os
import shutil

os.makedirs("data/raw", exist_ok=True)

# Dataset A — Customer Personality Analysis
kaggle_a = "/kaggle/input/customer-personality-analysis/marketing_campaign.csv"
local_a = "data/raw/marketing_campaign.csv"
if os.path.exists(kaggle_a) and not os.path.exists(local_a):
    os.symlink(kaggle_a, local_a)
    print(f"Linked: {kaggle_a} -> {local_a}")

# Dataset B — eCommerce (October)
kaggle_b_dir = "/kaggle/input/ecommerce-behavior-data-from-multi-category-store"
for fname in os.listdir(kaggle_b_dir):
    if fname.endswith(".csv"):
        src = os.path.join(kaggle_b_dir, fname)
        dst = os.path.join("data/raw", fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            print(f"Linked: {src} -> {dst}")

# %% Run Dataset A sweep (~1.5-2h)
"""
!python -m cadvae.eval.run_cadvae_sweep data=personality
"""

# %% Run Dataset B sweep (~5-6h)
"""
!python -m cadvae.eval.run_cadvae_sweep data=ecommerce eval.train_subsample=200000 model.max_epochs=40
"""

# %% Save results (download from Kaggle output)
"""
# Results are in results/phase5/. Kaggle auto-saves /kaggle/working/ as output.
# After the run, go to the notebook output tab and download the results/ folder.
!ls -la results/phase5/
!du -sh results/phase5/
"""
