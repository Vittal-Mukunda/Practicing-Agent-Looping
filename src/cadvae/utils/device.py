"""CUDA device verification (Phase 0 requirement).

Run:  python -m cadvae.utils.device

Prints a JSON device report and exits nonzero if CUDA is unavailable. A silent
CPU fallback is a known failure mode this project must catch (CLAUDE.md,
Hardware section), so ``resolve_device`` refuses to substitute CPU for a CUDA
request — CPU must be asked for explicitly.
"""

import json
from typing import Any

import torch


class CudaUnavailableError(RuntimeError):
    """Raised when CUDA is requested but unavailable (no silent CPU fallback)."""


def resolve_device(device: str = "cuda") -> torch.device:
    """Resolve the requested device string, refusing silent CPU fallback."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise CudaUnavailableError(
            "CUDA requested but torch.cuda.is_available() is False. Refusing "
            "silent CPU fallback; pass device=cpu explicitly if CPU is intended."
        )
    return torch.device(device)


def device_report() -> dict[str, Any]:
    """Collect device facts and run a small GPU op as evidence it computes."""
    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
    }
    if report["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        report["device_name"] = props.name
        report["vram_total_gb"] = round(props.total_memory / 1024**3, 2)
        report["compute_capability"] = f"{props.major}.{props.minor}"
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x).sum().item()  # forces actual kernel execution, not just allocation
        report["gpu_matmul_ok"] = y == y  # False would mean NaN
        report["vram_allocated_mb_after_matmul"] = round(
            torch.cuda.memory_allocated() / 1024**2, 1
        )
    return report


if __name__ == "__main__":
    rep = device_report()
    print(json.dumps(rep, indent=2))
    if not rep["cuda_available"]:
        raise SystemExit("FAIL: CUDA unavailable — PyTorch would silently run on CPU.")
