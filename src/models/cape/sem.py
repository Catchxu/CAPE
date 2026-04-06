from __future__ import annotations

from typing import Dict

from torch import Tensor, nn
import torch

from .modules import RMSNorm


def _make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class NonlinearLatentSem(nn.Module):
    """Nonlinear SEM operating in latent D-space."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mu_head = _make_mlp(embed_dim, hidden_dim, embed_dim, dropout=dropout)
        self.logvar_head = _make_mlp(embed_dim, hidden_dim, embed_dim, dropout=dropout)
        self.parent_phi = _make_mlp(embed_dim, hidden_dim, embed_dim, dropout=dropout)
        self.sem_psi = _make_mlp(2 * embed_dim, hidden_dim, embed_dim, dropout=dropout)
        self.decoder = _make_mlp(embed_dim, hidden_dim, 1, dropout=dropout)
        self.norm = RMSNorm(embed_dim)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor, adj: Tensor) -> Dict[str, Tensor]:
        # Nonlinear SEM is performed in latent D-space, not directly on raw X.
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        z = self.reparameterize(mu, logvar)

        parent_latent = self.parent_phi(z)
        parent_agg = torch.matmul(adj, parent_latent)
        z_recon = self.sem_psi(torch.cat([z, parent_agg], dim=-1))
        z_recon = self.norm(z_recon)
        x_recon = self.decoder(z_recon).squeeze(-1)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_recon": z_recon,
            "x_recon": x_recon,
        }
