"""CA-DVAE — Construct-Aligned Disentangled VAE (proposed model, Phase 4).

One model class; the two ablatable components are pure config points (CLAUDE.md
"two independently ablatable components"):

    L = recon_mse
        + beta               * KL_free       # Component 2 — disentanglement
        + aligned_kl_weight  * KL_aligned    # standard unit-Gaussian prior on named dims
        + lambda_align       * align_mse     # Component 1 — construct alignment

* **Component 1 — construct alignment (interpretability).** A designated block of
  ``aligned_dims`` latent dims is trained by a *linear* head to predict the named
  marketing constructs (RFM / price-sensitivity / category-affinity — Phase 2).
  Linear (not MLP) so each construct maps back to an interpretable direction in
  latent space and the per-axis alignment score (Phase 6) is well defined.
* **Component 2 — disentanglement.** beta-VAE KL weighting on the *remaining free*
  dims (CLAUDE.md). Aligned dims keep a standard unit-Gaussian prior
  (``aligned_kl_weight``, default 1.0) so they stay stochastic/regularized rather
  than collapsing under the alignment pressure.

Ablations (Phase 4/5) are config points, not separate code:
    * beta-VAE  (proposed - alignment)       : lambda_align = 0  (-> aligned_dims
                                               forced 0, beta weights the whole KL)
    * VAE+align (proposed - disentanglement) : beta = 1.0
    * CA-DVAE full                           : beta swept, lambda_align > 0

``beta`` and ``lambda_align`` are the Phase-5 swept variables that generate the
interpretability-performance trade-off curve.

Reconstruction is Gaussian / MSE (features are standardized-continuous, matching the
AE baseline so lift differences are about the *objective*, not the likelihood model).
Encoder/decoder share the AE's dense backbone shape (same ``latent_dim`` /
``hidden_dims``) for the same reason. Trivially fits 6 GB VRAM (dense MLP over <=34
features).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from cadvae.utils.device import resolve_device
from cadvae.utils.seeding import seed_everything


def resolve_aligned_dims(model_cfg, n_constructs: int) -> int:
    """Number of construct-aligned latent dims (D-021).

    ``aligned_dims=null`` -> auto: ``min(n_constructs, latent_dim - min_free_dims)``,
    so at least ``min_free_dims`` free dims remain for the disentanglement block.
    Forced to 0 when ``lambda_align == 0`` (the beta-VAE ablation is a *pure*
    beta-VAE: every dim is free and beta weights the whole KL).
    """
    latent = int(model_cfg.latent_dim)
    if float(model_cfg.lambda_align) == 0.0:
        return 0
    a = model_cfg.get("aligned_dims")
    if a is None:
        min_free = int(model_cfg.get("min_free_dims", 4))
        a = min(int(n_constructs), latent - min_free)
    a = int(a)
    if not 0 <= a <= latent:
        raise ValueError(f"aligned_dims={a} out of range [0, {latent}] (latent_dim)")
    if a > 0 and n_constructs <= 0:
        raise ValueError("aligned_dims > 0 requires construct targets (n_constructs > 0)")
    return a


class CADVAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dims: list[int],
                 aligned_dims: int, n_constructs: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.aligned_dims = int(aligned_dims)

        enc: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            enc += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.enc_trunk = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(d, self.latent_dim)
        self.fc_logvar = nn.Linear(d, self.latent_dim)

        dec: list[nn.Module] = []
        d = self.latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec.append(nn.Linear(d, in_dim))
        self.decoder = nn.Sequential(*dec)

        # linear alignment head over the aligned latent block (None if no aligned dims)
        self.align_head: nn.Linear | None = (
            nn.Linear(self.aligned_dims, int(n_constructs)) if self.aligned_dims > 0 else None
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z)
        align_pred = (self.align_head(z[:, : self.aligned_dims])
                      if self.align_head is not None else None)
        return {"xhat": xhat, "mu": mu, "logvar": logvar, "z": z, "align_pred": align_pred}


def _kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-dim KL of N(mu, var) from N(0, 1): 0.5*(mu^2 + var - logvar - 1) (>= 0)."""
    return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)


def cadvae_loss(out: dict, x: torch.Tensor, C: torch.Tensor | None, aligned_dims: int,
                beta: float, lambda_align: float, aligned_kl_weight: float) -> dict:
    """ELBO-scaled loss (recon & KL summed over their dimension, averaged over batch)."""
    recon = F.mse_loss(out["xhat"], x, reduction="none").sum(1).mean()
    kld = _kl_per_dim(out["mu"], out["logvar"])                      # (B, latent)
    kl_aligned = (kld[:, :aligned_dims].sum(1).mean()
                  if aligned_dims > 0 else x.new_zeros(()))
    kl_free = kld[:, aligned_dims:].sum(1).mean()                    # = all dims when aligned == 0
    if out["align_pred"] is not None and C is not None:
        align = F.mse_loss(out["align_pred"], C, reduction="none").sum(1).mean()
    else:
        align = x.new_zeros(())
    total = recon + beta * kl_free + aligned_kl_weight * kl_aligned + lambda_align * align
    return {"total": total, "recon": recon, "kl_free": kl_free,
            "kl_aligned": kl_aligned, "align": align}


def train_cadvae(X: np.ndarray, C: np.ndarray | None, model_cfg, seed: int,
                 device: str = "cuda", n_constructs: int | None = None,
                 log_fn=None) -> tuple[CADVAE, list[dict]]:
    """Train the CA-DVAE. Deterministic given ``seed``. Returns (model, loss history).

    ``C`` is the standardized construct-target matrix (alignment labels); may be None
    only for the beta-VAE ablation (lambda_align == 0).
    """
    seed_everything(seed)
    dev = resolve_device(device)   # raises if cuda requested but unavailable — D-027
    if n_constructs is None:
        n_constructs = 0 if C is None else int(C.shape[1])
    aligned = resolve_aligned_dims(model_cfg, n_constructs)
    if aligned > 0 and C is None:
        raise ValueError("CA-DVAE alignment is active (aligned_dims > 0) but C is None")

    Xt = torch.as_tensor(X, dtype=torch.float32)
    use_C = aligned > 0 and C is not None
    tensors = (Xt, torch.as_tensor(C, dtype=torch.float32)) if use_C else (Xt,)
    loader = DataLoader(
        TensorDataset(*tensors), batch_size=int(model_cfg.batch_size), shuffle=True,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )
    model = CADVAE(X.shape[1], int(model_cfg.latent_dim), list(model_cfg.hidden_dims),
                   aligned, n_constructs).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(model_cfg.lr))
    beta = float(model_cfg.beta)
    lam = float(model_cfg.lambda_align)
    akw = float(model_cfg.get("aligned_kl_weight", 1.0))

    history: list[dict] = []
    model.train()
    for epoch in range(int(model_cfg.max_epochs)):
        agg: dict[str, float] = defaultdict(float)
        nb = 0
        for batch in loader:
            xb = batch[0].to(dev)
            cb = batch[1].to(dev) if len(batch) > 1 else None
            opt.zero_grad()
            out = model(xb)
            losses = cadvae_loss(out, xb, cb, aligned, beta, lam, akw)
            losses["total"].backward()
            opt.step()
            for k, v in losses.items():
                agg[k] += float(v.detach())
            nb += 1
        entry = {"epoch": epoch, **{k: agg[k] / nb for k in agg}}
        history.append(entry)
        if log_fn is not None:
            log_fn(entry, step=epoch)
    model.aligned_dims_used = aligned  # type: ignore[assignment]  # informational
    return model, history


@torch.no_grad()
def encode(model: CADVAE, X: np.ndarray, device: str = "cuda",
           batch_size: int = 8192) -> np.ndarray:
    """Frozen encode -> posterior-mean latent embedding mu (all dims), eval mode."""
    dev = next(model.parameters()).device
    model.eval()
    out = []
    Xt = torch.as_tensor(X, dtype=torch.float32)
    for i in range(0, len(Xt), batch_size):
        mu, _ = model.encode(Xt[i:i + batch_size].to(dev))
        out.append(mu.cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def align_predict(model: CADVAE, X: np.ndarray, device: str = "cuda",
                  batch_size: int = 8192) -> np.ndarray | None:
    """Alignment-head construct predictions from the frozen aligned latent block (mu).

    Uses the deterministic posterior mean (no sampling) — appropriate for frozen eval.
    Returns None for the beta-VAE ablation (no alignment head).
    """
    if model.align_head is None:
        return None
    dev = next(model.parameters()).device
    model.eval()
    a = model.aligned_dims
    out = []
    Xt = torch.as_tensor(X, dtype=torch.float32)
    for i in range(0, len(Xt), batch_size):
        mu, _ = model.encode(Xt[i:i + batch_size].to(dev))
        out.append(model.align_head(mu[:, :a]).cpu().numpy())
    return np.concatenate(out, axis=0)
