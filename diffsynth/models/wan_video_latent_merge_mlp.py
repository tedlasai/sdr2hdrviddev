import torch
import torch.nn as nn
from einops import rearrange


class LatentMergeMLP(nn.Module):
    """
    Per-cell MLP that blends E exposure latents into one merged latent.

    Each spatial-temporal cell has E latent vectors (one per exposure bracket).
    A shared encoder maps each to a hidden representation, a weight head
    produces per-exposure softmax weights, and a projection head outputs
    the merged latent vector.

    Input:  (B, E, C, T', H', W') — stacked latents, one per exposure
    Output: (B, C, T', H', W')   — single merged latent for VAE decoding
    """

    def __init__(self, latent_dim: int = 48, hidden_dim: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )
        self.weight_head = nn.Linear(hidden_dim, 1)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        B, E, C, T, H, W = latents.shape
        # Flatten spatial+temporal into batch: (N, E, C) where N = B*T'*H'*W'
        x = rearrange(latents.float(), 'b e c t h w -> (b t h w) e c')
        h = self.enc(x)                         # (N, E, hidden_dim)
        logits = self.weight_head(h)            # (N, E, 1)
        weights = torch.softmax(logits, dim=1)  # (N, E, 1)
        merged_h = (weights * h).sum(dim=1)     # (N, hidden_dim)
        merged = self.out_proj(merged_h)        # (N, C)
        out = rearrange(merged, '(b t h w) c -> b c t h w', b=B, t=T, h=H, w=W)
        return out.to(latents.dtype)
