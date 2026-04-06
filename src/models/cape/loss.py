from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def elbo_loss(
    x: Tensor,
    x_recon: Tensor,
    mu: Tensor,
    logvar: Tensor,
    reduction: str = "mean",
    beta: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    recon = F.mse_loss(x_recon, x, reduction="none").sum(dim=-1)
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(-1, -2))

    if reduction == "mean":
        recon_loss = recon.mean()
        kl_loss = kl.mean()
    elif reduction == "sum":
        recon_loss = recon.sum()
        kl_loss = kl.sum()
    elif reduction == "none":
        recon_loss = recon
        kl_loss = kl
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    elbo = recon_loss + beta * kl_loss
    return elbo, recon_loss, kl_loss


def graph_sparsity_loss(adj: Tensor, reduction: str = "mean") -> Tensor:
    sparse = adj.abs()
    if reduction == "mean":
        return sparse.mean()
    if reduction == "sum":
        return sparse.sum()
    if reduction == "none":
        return sparse
    raise ValueError(f"Unsupported reduction: {reduction}")
