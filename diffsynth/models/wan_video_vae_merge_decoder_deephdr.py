import torch
import torch.nn as nn
import sys
sys.path.append("../../")
from utils import merge_hdr, merge_hdr_avg_linear_boost  # keep if you need it
from einops import rearrange


class DeepHDRCNN(nn.Module):
    """
    Weight-estimation trunk from Kalantari & Ramamoorthi, "Deep High Dynamic Range
    Imaging of Dynamic Scenes" (SIGGRAPH 2017) — https://github.com/CharlieMarcotte/Deep-HDR-with-Pytorch.
    Same 4-conv-layer architecture (7x7 -> 5x5 -> 3x3 -> 1x1, channels 18-100-100-50-9),
    but convolutions use 'same' padding instead of the original 'valid' padding so the
    output resolution matches the input frame (the reference implementation trains on
    small patches and crops the target to the shrunk 'valid' output instead).
    """
    def __init__(self, in_channels=18, out_channels=9):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 100, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(100, 50, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Conv2d(50, out_channels, kernel_size=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class WanVideoVAEMergeDecoderDeepHDR(nn.Module):
    """
    Pixel-space merger built on the DeepHDR weight-estimation CNN. For each exposure,
    builds a 6-channel [ldr_rgb, radiance_rgb] stack (radiance = ldr * 2**(-EV), matching
    the convention used by the other mergers in this codebase); the per-exposure stacks
    are concatenated (18 channels for 3 exposures) and fed through `DeepHDRCNN` to predict
    per-exposure, per-channel blend-weight logits. Weights are softmax-normalized over the
    exposure axis (rather than the reference repo's raw-sum normalization) and used to
    blend the radiance images, matching the softmax-weighted-merge convention already used
    by `WanVideoVAEMergeDecoder`/`WanVideoVAEMergeDecoderMLP`.
    """
    def __init__(self, num_exposures=3):
        super().__init__()
        self.num_exposures = num_exposures
        self.net = DeepHDRCNN(in_channels=6 * num_exposures, out_channels=3 * num_exposures)

    @torch.no_grad()
    def _check(self, vids):
        assert vids.dim() == 6, f"expected (B,E,C,T,H,W), got {tuple(vids.shape)}"
        B, E, C, T, H, W = vids.shape
        assert C == 3, "expects RGB"
        return B, E, C, T, H, W

    def _stack_inputs(self, videos):
        if isinstance(videos, (list, tuple)):
            vids = torch.stack(videos, dim=1)  # (B,E,3,T,H,W)
        else:
            vids = videos
        return vids

    def _merge_frame(self, ldr_t, radiance_t):
        # ldr_t, radiance_t: (B, E, C, H, W)
        B, E, C, H, W = ldr_t.shape
        x = torch.cat([
            rearrange(ldr_t, 'b e c h w -> b (e c) h w'),
            rearrange(radiance_t, 'b e c h w -> b (e c) h w'),
        ], dim=1)  # (B, 6E, H, W)
        logits = self.net(x)  # (B, 3E, H, W)
        logits = rearrange(logits, 'b (e c) h w -> b e c h w', e=E, c=C)
        weights = torch.softmax(logits, dim=1)  # softmax over exposures
        return (weights * radiance_t).sum(dim=1)  # (B, C, H, W)

    def forward(
        self,
        videos,
        exposures,
        encoder_decoder_mode,
        mem_efficient: bool = False,
    ) -> torch.Tensor:
        vids = self._stack_inputs(videos)  # (B,E,3,T,H,W)
        B, E, C, T, H, W = self._check(vids)

        if not torch.is_tensor(exposures):
            exposures = torch.tensor(exposures, device=vids.device, dtype=vids.dtype)
        exposures = exposures.to(device=vids.device, dtype=vids.dtype)

        # Canonicalize to ascending-EV order. The conv trunk below concatenates all
        # exposures into fixed channel slots (unlike the other mergers in this codebase,
        # which tag each exposure's channels with its EV value), so its learned weights
        # are tied to a specific exposure ordering. Callers are not guaranteed to agree on
        # an order (e.g. training data loaders vs. ValScheduler's `sorted(self.exposures)`
        # at eval time), so we enforce one here rather than trusting the caller.
        order = torch.argsort(exposures)
        vids = vids[:, order]
        exposures = exposures[order]

        # Special-case: classic merge override (matches the other mergers in this codebase)
        if encoder_decoder_mode == "seperate_debevec":
            assert E == 3, "seperate_debevec expects exactly 3 exposures"
            radiance = vids * (2.0 ** (-exposures)).view(1, E, 1, 1, 1, 1)
            low_idx = (exposures == -4).nonzero(as_tuple=True)[0].item()
            normal_idx = (exposures == 0).nonzero(as_tuple=True)[0].item()
            high_idx = (exposures == 4).nonzero(as_tuple=True)[0].item()
            normal, low, high = vids[:, normal_idx], vids[:, low_idx], vids[:, high_idx]
            normal_r, low_r, high_r = radiance[:, normal_idx], radiance[:, low_idx], radiance[:, high_idx]
            return merge_hdr(normal, low, high, normal_r, low_r, high_r)

        assert E == self.num_exposures, f"expected {self.num_exposures} exposures, got {E}"


        print("EXPOSURES: ", exposures)
        radiance = vids * (2.0 ** (-exposures)).view(1, E, 1, 1, 1, 1)  # (B,E,C,T,H,W)

        if not mem_efficient:
            ldr_t = rearrange(vids, 'b e c t h w -> (b t) e c h w')
            rad_t = rearrange(radiance, 'b e c t h w -> (b t) e c h w')
            fused = self._merge_frame(ldr_t, rad_t)  # (B*T, C, H, W)
            out = rearrange(fused, '(b t) c h w -> b c t h w', b=B, t=T)
        else:
            out = vids.new_empty((B, C, T, H, W))
            for t in range(T):
                fused_t = self._merge_frame(vids[:, :, :, t], radiance[:, :, :, t])  # (B, C, H, W)
                out[:, :, t] = fused_t

        return out
