"""Seed-control tests: same seed → identical draws; different seed → different."""

import numpy as np
import pytest
import torch

from cadvae.utils.seeding import seed_everything


def test_same_seed_reproduces_numpy_and_torch():
    seed_everything(123)
    a1, t1 = np.random.rand(8), torch.randn(8)
    seed_everything(123)
    a2, t2 = np.random.rand(8), torch.randn(8)
    assert np.array_equal(a1, a2)
    assert torch.equal(t1, t2)


def test_different_seeds_differ():
    seed_everything(1)
    t1 = torch.randn(128)
    seed_everything(2)
    t2 = torch.randn(128)
    assert not torch.equal(t1, t2)


def test_cuda_rng_reproducible():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable (already failed loudly in test_environment)")
    seed_everything(7)
    g1 = torch.randn(64, device="cuda")
    seed_everything(7)
    g2 = torch.randn(64, device="cuda")
    assert torch.equal(g1, g2)
