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
from threadpoolctl import threadpool_limits
from torch import nn

from cadvae.models.ae import Autoencoder, encode
from cadvae.utils.seeding import seed_everything


def _soft_assign(z: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    # ||z-mu||^2 via the matmul expansion, NOT torch.cdist: cdist's CUDA backward uses
    # nondeterministic atomicAdd (pytorch.org/docs/stable/notes/randomness.html), which
    # made DEC the single non-reproducible baseline (caught by an identity re-run;
    # every other method was bit-identical). Matmuls are deterministic under
    # CUBLAS_WORKSPACE_CONFIG (set by seed_everything).
    dist2 = (z * z).sum(1, keepdim=True) + (centroids * centroids).sum(1) \
        - 2.0 * (z @ centroids.t())                 # (N, K)
    dist2 = dist2.clamp_min(0.0)                    # guard tiny negative rounding
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
    # Single-threaded fit: multithreaded-BLAS reductions make sklearn's
    # cluster_centers_ wobble by ~1e-7 across bit-identical inputs (measured on this
    # repo's env: two OpenBLAS builds, 16 threads). Discrete baselines absorb that ulp
    # noise, but DEC uses the centers as CONTINUOUS init and training amplifies it
    # chaotically — DEC was the single non-reproducible baseline (caught by identity
    # re-runs). threadpool_limits(1) makes the reduction order, hence the fit,
    # bit-reproducible; the fit is tiny relative to DEC training itself. (D-028)
    with threadpool_limits(limits=1):
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
        # permute on CPU (deterministic), then move: CUDA randperm is another
        # nondeterminism source under warn-only deterministic mode
        gen = torch.Generator().manual_seed(seed + it)
        perm = torch.randperm(len(Xt), generator=gen).to(dev)
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
