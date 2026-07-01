import torch
import torch.nn as nn
from einops import rearrange


class WanVideoVAELatentMerger(nn.Module):
    """
    Merges per-exposure latents into a single latent with a per-voxel MLP (1x1x1 convs),
    applied before VAE decoding rather than merging decoded pixels afterward. The MLP
    predicts a per-exposure, per-channel blend weight at every voxel; weights are
    softmax-normalized over the exposure axis and used to blend the input latents
    (mirrors the softmax-weighted-radiance merge used by the pixel-space mergers). The
    VAE decoder is expected to map the merged latent to log-HDR pixel space; callers
    must exponentiate the decoder output to recover linear HDR.
    """
    def __init__(self, latent_channels, num_exposures=3, hidden_dim=256, num_layers=4):
        super().__init__()
        assert num_layers >= 2
        self.latent_channels = latent_channels
        self.num_exposures = num_exposures

        in_ch = latent_channels * num_exposures + num_exposures  # stacked latents + per-exposure EV channel
        out_ch = latent_channels * num_exposures  # per-exposure, per-channel blend weight logits
        layers = [nn.Conv3d(in_ch, hidden_dim, kernel_size=1), nn.LeakyReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1), nn.LeakyReLU(inplace=True)]
        layers.append(nn.Conv3d(hidden_dim, out_ch, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, latents, exposures):
        # latents: (B, E, C, T, H, W)
        B, E, C, T, H, W = latents.shape
        assert E == self.num_exposures, f"expected {self.num_exposures} exposures, got {E}"
        assert C == self.latent_channels, f"expected {self.latent_channels} latent channels, got {C}"

        if not torch.is_tensor(exposures):
            exposures = torch.tensor(exposures, device=latents.device, dtype=latents.dtype)
        exposures = exposures.to(device=latents.device, dtype=latents.dtype)
        exp_map = exposures.view(1, E, 1, 1, 1).expand(B, E, T, H, W)

        x = torch.cat([
            rearrange(latents, 'b e c t h w -> b (e c) t h w'),
            exp_map,
        ], dim=1)
        logits = self.net(x)  # (B, E*C, T, H, W)
        logits = rearrange(logits, 'b (e c) t h w -> b e c t h w', e=E, c=C)
        weights = torch.softmax(logits, dim=1)  # softmax over exposures
        merged = (weights * latents).sum(dim=1)  # (B, C, T, H, W)
        return merged
