# CA-DVAE Phase-5 sweep — Kaggle notebook cells (VERIFIED against the repo code).
#
# Paste each CELL below into its own Kaggle notebook cell, in order.
# Full setup instructions: cloud/kaggle/README.md
#
# Design:
# * No raw data needed — the sweep reads only the two processed parquet caches,
#   uploaded as a private Kaggle dataset (see README).
# * polars is pinned EXACTLY: dataset B's train/val/test split is a seed-salted
#   polars hash of user_id, and polars hashes are not stable across versions.
#   A version drift would silently produce splits that do not match the local
#   Phase-3 baseline bars. sklearn pinned for identical downstream heads.
# * Results auto-push to the `kaggle-results` GitHub branch every 15 min and
#   after each sweep; a fresh session restores from that branch first, so the
#   sweep's per-run resume-skip continues across Save & Run All rounds.

# %% ===== CELL 1: clone + pinned install + GPU check =====
import os
import subprocess
import sys

REPO_URL = "https://github.com/Vittal-Mukunda/Practicing-Agent-Looping.git"
WORK = "/kaggle/working/cadvae"

if not os.path.exists(WORK):
    subprocess.run(["git", "clone", REPO_URL, WORK], check=True)
os.chdir(WORK)
os.environ["PYTHONPATH"] = f"{WORK}/src"

# Pin split- and head-determining packages to the exact local versions (uv.lock).
# Do NOT touch torch/numpy: Kaggle's preinstalled torch is built for its GPUs.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "polars==1.42.1", "scikit-learn==1.9.0",
     "hydra-core==1.3.4", "omegaconf==2.3.1"],
    check=True,
)

import polars
import sklearn
import torch

assert torch.cuda.is_available(), "No GPU — set Accelerator to GPU P100 in Settings"
print("python :", sys.version.split()[0])
print("torch  :", torch.__version__, "|", torch.cuda.get_device_name(0))
print("polars :", polars.__version__, "| sklearn:", sklearn.__version__)
assert polars.__version__ == "1.42.1", "polars pin failed — splits would not match local"
assert sklearn.__version__ == "1.9.0", "sklearn pin failed — heads would not match local"

# %% ===== CELL 2: processed caches from the private dataset =====
import glob
import shutil

def find_one(name: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert len(hits) == 1, f"{name}: expected exactly 1 match under /kaggle/input, got {hits}"
    return hits[0]

os.makedirs("data/processed/personality", exist_ok=True)
os.makedirs("data/processed/ecommerce", exist_ok=True)
shutil.copy(find_one("engineered.parquet"), "data/processed/personality/engineered.parquet")
shutil.copy(find_one("user_table.parquet"), "data/processed/ecommerce/user_table.parquet")

import polars as pl

a_shape = pl.read_parquet("data/processed/personality/engineered.parquet").shape
b_rows = pl.scan_parquet("data/processed/ecommerce/user_table.parquet") \
    .select(pl.len()).collect().item()
print("A engineered:", a_shape)      # expect (2240, 27)
print("B user_table rows:", b_rows)  # expect ~2,550,000
assert a_shape[0] == 2240 and b_rows > 2_000_000, "cache sanity check failed"

# %% ===== CELL 3: GitHub persistence — restore previous progress + autopush =====
import threading
import time

from kaggle_secrets import UserSecretsClient

TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
AUTH_URL = f"https://Vittal-Mukunda:{TOKEN}@github.com/Vittal-Mukunda/Practicing-Agent-Looping.git"
RES_REPO = "/tmp/results_repo"   # /tmp is NOT saved into notebook output -> token cannot leak
BRANCH = "kaggle-results"

def _git(*args, ok_fail=False):
    r = subprocess.run(["git", *args], cwd=RES_REPO, capture_output=True, text=True)
    if r.returncode != 0 and not ok_fail:
        raise RuntimeError(f"git {args[0]} failed: {(r.stderr + r.stdout).replace(TOKEN, '***')}")
    return r.returncode

if not os.path.exists(RES_REPO):
    r = subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch",
                        AUTH_URL, RES_REPO], capture_output=True, text=True)
    if r.returncode != 0:                       # branch doesn't exist yet -> start it
        os.makedirs(RES_REPO)
        _git("init", "-b", BRANCH)
        _git("remote", "add", "origin", AUTH_URL)
    _git("config", "user.email", "vittal.muku@gmail.com")
    _git("config", "user.name", "Kaggle Autosave")

# RESTORE: completed runs from previous sessions -> results/, so resume-skip continues
restored = 0
if os.path.exists(f"{RES_REPO}/phase5"):
    for root, _, files in os.walk(f"{RES_REPO}/phase5"):
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join("results/phase5", os.path.relpath(src, f"{RES_REPO}/phase5"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                restored += 1
print(f"restored {restored} files from previous sessions")

def push_results(msg: str) -> None:
    """Copy results -> results repo, commit, push. NEVER raises into the sweep."""
    try:
        if os.path.exists("results/phase5"):
            shutil.copytree("results/phase5", f"{RES_REPO}/phase5", dirs_exist_ok=True)
        _git("add", "-A")
        if _git("commit", "-m", msg, ok_fail=True) == 0:    # nonzero = nothing new
            _git("push", "-u", "origin", BRANCH)
            print(f"[autosave] pushed: {msg}", flush=True)
    except Exception as e:  # noqa: BLE001 — autosave must never kill the sweep
        print(f"[autosave] FAILED (sweep unaffected): {e}", flush=True)

def _autopush_loop():
    while True:
        time.sleep(900)          # every 15 min
        push_results("autosave")

threading.Thread(target=_autopush_loop, daemon=True).start()
push_results("session start")
print("GitHub persistence ready (branch: kaggle-results)")

# %% ===== CELL 4: Dataset A sweep (144 runs, ~1.5-2.5 h) =====
def sh(*args):
    """Run a command with live streamed output; raise on failure."""
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"command failed with exit {p.returncode}")

sh(sys.executable, "-m", "cadvae.eval.run_cadvae_sweep", "data=personality")
push_results("dataset A sweep complete")

# %% ===== CELL 5: Dataset B sweep (144 runs, ~7-11 h — may need a 2nd round) =====
sh(sys.executable, "-m", "cadvae.eval.run_cadvae_sweep", "data=ecommerce",
   "eval.train_subsample=200000", "model.max_epochs=40")
push_results("dataset B sweep complete")

# %% ===== CELL 6: final verification =====
n_a = len(glob.glob("results/phase5/personality_sweep/runs/*.json"))
n_b = len(glob.glob("results/phase5/ecommerce_sweep/runs/*.json"))
print(f"personality: {n_a}/144   ecommerce: {n_b}/144")
push_results("final results")
assert n_a == 144 and n_b == 144, \
    "INCOMPLETE — progress is saved on GitHub; just Save & Run All again to resume"
print("ALL 288 RUNS COMPLETE AND PUSHED (branch: kaggle-results)")
