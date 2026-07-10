"""Dense autoencoder — the hard-baseline representation (Phase 3) and the encoder
backbone reused by AE+K-means, GMM-on-AE, and DEC.

Kept architecturally comparable to the CA-DVAE (same ``latent_dim`` / ``hidden_dims``)
so downstream-lift comparisons are about the *objective*, not capacity. Trivially
fits 6 GB VRAM (dense MLP over ≤34 features).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cadvae.utils.device import resolve_device
from cadvae.utils.seeding import seed_everything


class Autoencoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        enc: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            enc += [nn.Linear(d, h), nn.ReLU()]
            d = h
        enc.append(nn.Linear(d, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        d = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec.append(nn.Linear(d, in_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(X: np.ndarray, model_cfg, seed: int, device: str = "cuda") -> Autoencoder:
    """Train an AE with MSE reconstruction. Deterministic given ``seed``."""
    seed_everything(seed)
    dev = resolve_device(device)   # raises if cuda requested but unavailable — D-027
    Xt = torch.as_tensor(X, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(Xt), batch_size=int(model_cfg.batch_size), shuffle=True,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )
    model = Autoencoder(X.shape[1], int(model_cfg.latent_dim), list(model_cfg.hidden_dims)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(model_cfg.lr))
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(int(model_cfg.max_epochs)):
        for (xb,) in loader:
            xb = xb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def encode(model: Autoencoder, X: np.ndarray, device: str = "cuda",
           batch_size: int = 8192) -> np.ndarray:
    """Frozen encode → latent embeddings (no grad, eval mode)."""
    dev = next(model.parameters()).device
    model.eval()
    out = []
    Xt = torch.as_tensor(X, dtype=torch.float32)
    for i in range(0, len(Xt), batch_size):
        out.append(model.encoder(Xt[i:i + batch_size].to(dev)).cpu().numpy())
    return np.concatenate(out, axis=0)
