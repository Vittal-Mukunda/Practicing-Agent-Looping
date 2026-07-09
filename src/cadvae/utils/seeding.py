"""Global seed control.

Reference: https://pytorch.org/docs/stable/notes/randomness.html

Known limits (documented per CLAUDE.md rather than silently ignored): some CUDA
kernels have no deterministic implementation. We (a) seed every RNG, (b) force
deterministic cuDNN kernels, (c) enable ``torch.use_deterministic_algorithms``
in warn-only mode so any nondeterministic op is *surfaced in logs* instead of
passing silently. Residual GPU nondeterminism, if observed, is absorbed by the
multi-seed reporting protocol (mean ± std over ≥5 seeds).
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and torch (CPU + all CUDA devices).

    With ``deterministic=True`` also configures cuDNN for deterministic
    kernels and sets ``CUBLAS_WORKSPACE_CONFIG`` (required by CUDA >= 10.2
    for deterministic CUBLAS matmuls) if the caller has not set it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU and all CUDA devices
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: nondeterministic ops emit a warning (visible in run logs)
        # instead of raising — revisit at Phase 4 once the training loop exists.
        torch.use_deterministic_algorithms(True, warn_only=True)
