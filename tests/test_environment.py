"""Phase 0 hardware verification.

The CUDA test is a HARD failure, not a skip: this project's protocol requires
the RTX 4050, and a silent CPU fallback is the exact failure mode CLAUDE.md
says to catch. If these fail, fix the environment — do not weaken the test.
"""

import pytest
import torch

from cadvae.utils.device import CudaUnavailableError, device_report, resolve_device


def test_cuda_available_and_computing():
    rep = device_report()
    assert rep["cuda_available"], (
        "torch.cuda.is_available() is False — GPU required (CLAUDE.md, Hardware). "
        f"torch={rep['torch_version']}, cuda_runtime={rep['cuda_runtime_version']}"
    )
    assert rep["gpu_matmul_ok"], "GPU matmul produced NaN"
    assert rep["vram_total_gb"] > 0


def test_resolve_device_refuses_silent_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(CudaUnavailableError):
        resolve_device("cuda")


def test_resolve_device_explicit_cpu_allowed():
    assert resolve_device("cpu").type == "cpu"
