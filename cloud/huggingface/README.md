# HuggingFace Spaces Setup

Paid GPU only ($0.60/hr for T4). More complex than Kaggle/Colab.

**Honest note:** HF Spaces is designed for hosting apps (Gradio/Streamlit), not
batch training. It works, but setup is heavier and you're paying for something
Kaggle gives free. Consider Kaggle first.

## Cost estimate

- T4 ($0.60/hr) × ~10 hours = **~$6**
- A10G ($1.05/hr) × ~7 hours = **~$7** (faster GPU)

## Steps (2-3 hours first time)

### 1. Create a HuggingFace Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name: `cadvae-sweep`
3. SDK: **Docker**
4. Hardware: **T4 small** (or A10G if budget allows)
5. Visibility: **Private**

### 2. Upload the Dockerfile and run script

Copy `Dockerfile` and `run_sweep.sh` from this folder into the Space repo:

```bash
# Clone your HF space
git clone https://huggingface.co/spaces/YOUR_USERNAME/cadvae-sweep
cd cadvae-sweep

# Copy files
cp /path/to/cloud/huggingface/Dockerfile .
cp /path/to/cloud/huggingface/run_sweep.sh .

# You need to set your Kaggle credentials as Space secrets:
# Go to Space Settings → Variables and secrets → New secret
#   KAGGLE_USERNAME = your_username
#   KAGGLE_KEY = your_api_key

git add -A && git commit -m "Add sweep runner" && git push
```

### 3. Monitor and retrieve results

- The Space logs show sweep progress
- Results are written to the Space's persistent storage
- Download via the HF Hub API or the Files tab

### 4. After the sweep

Download `results/phase5/` and copy to your local machine:

```bash
# Using huggingface_hub
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('spaces/YOUR_USERNAME/cadvae-sweep', local_dir='./hf_results', allow_patterns='results/*')
"
```

Then run Phase 6 locally:
```bash
.venv\Scripts\python.exe -m cadvae.eval.run_phase6
```
