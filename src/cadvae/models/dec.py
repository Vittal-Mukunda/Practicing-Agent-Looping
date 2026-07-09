"""Deep Embedded Clustering (DEC, Xie et al., ICML 2016) — hard neural baseline.

Pipeline: reuse the pretrained AE encoder, initialize K cluster centroids via
K-means on the AE embedding, then jointly fine-tune the encoder + centroids to
minimize KL(P‖Q) between the Student-t soft assignment Q and its sharpened target P.

Soft assignment (α = 1 degrees of freedom):
    q_ij = (1 + ||z_i − μ_j||² )^(−1) / Σ_k (1 + ||z_i − μ_k||²)^(−1)
Target:
    p_ij = (q_ij² / Σ_i q_ij) / Σ_k (q_ik² / Σ_i q_ik)
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn

from cadvae.models.ae import Autoencoder, encode
from cadvae.utils.seeding import seed_everything


def _soft_assign(z: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    dist2 = torch.cdist(z, centroids) ** 2          # (N, K)
    q = 1.0 / (1.0 + dist2)
    return q / q.sum(dim=1, keepdim=True)


def _target_distribution(q: torch.Tensor) -> torch.Tensor:
    weight = q ** 2 / q.sum(dim=0)
    return (weight.t() / weight.sum(dim=1)).t()


def train_dec(ae: Autoencoder, X: np.ndarray, model_cfg, n_clusters: int, seed: int,
              device: str = "cuda", max_iters: int = 40, update_interval: int = 5) -> dict:
    """Fine-tune encoder + centroids from an AE init. Returns encoder + centroids."""
    seed_everything(seed)
    dev = next(ae.parameters()).device
    encoder = ae.encoder

    z0 = encode(ae, X, device=device)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(z0)
    centroids = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=dev))

    opt = torch.optim.Adam([*encoder.parameters(), centroids], lr=float(model_cfg.lr))
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    bs = int(model_cfg.batch_size)
    encoder.train()
    for it in range(max_iters):
        if it % update_interval == 0:
            with torch.no_grad():
                q_all = _soft_assign(encoder(Xt), centroids)
                p_all = _target_distribution(q_all).detach()
        gen = torch.Generator(device=dev).manual_seed(seed + it)
        perm = torch.randperm(len(Xt), generator=gen, device=dev)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            q = _soft_assign(encoder(Xt[idx]), centroids)
            loss = nn.functional.kl_div(q.log(), p_all[idx], reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {"encoder": encoder, "centroids": centroids.detach()}


@torch.no_grad()
def dec_assign(dec: dict, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Soft cluster assignment Q (N × K) for frozen DEC."""
    encoder, centroids = dec["encoder"], dec["centroids"]
    dev = centroids.device
    encoder.eval()
    out = []
    Xt = torch.as_tensor(X, dtype=torch.float32)
    for i in range(0, len(Xt), batch_size):
        z = encoder(Xt[i:i + batch_size].to(dev))
        out.append(_soft_assign(z, centroids).cpu().numpy())
    return np.concatenate(out, axis=0)
