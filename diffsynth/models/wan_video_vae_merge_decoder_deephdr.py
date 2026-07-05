import math
import torch
import torch.nn as nn
from einops import rearrange

# mu-law tonemapping constants, matching Constants.mu in the original
# Deep-HDR-with-Pytorch repo. HDR_MAX is the largest linear radiance value this
# merger can represent: with exposures fixed to (-4, 0, 4), the darkest (-4 EV)
# exposure has radiance = ldr * 2**4, so a fully-saturated pixel there tops out
# at 16 — that's the domain the mu-law curve is stretched over.
MU = 5000.0
HDR_MAX = 16.0


def mu_law_compress(x, mu=MU, hdr_max=HDR_MAX):
    """Linear HDR radiance -> tonemapped [0,1) domain."""
    x = (x / hdr_max).clamp(min=0.0)
    return torch.log1p(mu * x) / math.log1p(mu)


def mu_law_expand(y, mu=MU, hdr_max=HDR_MAX):
    """Tonemapped [0,1) domain -> linear HDR radiance (inverse of mu_law_compress)."""
    y = y.clamp(min=0.0, max=1.0 - 1e-6)
    return hdr_max * torch.expm1(y * math.log1p(mu)) / mu


class WanVideoVAEMergeDecoderDeepHDR(nn.Module):
    """
    Direct-CNN HDR merger, replicating the "Direct" architecture from
    Kalantari & Ramamoorthi, "Deep High Dynamic Range Imaging of Dynamic Scenes"
    (SIGGRAPH 2017), as implemented in
    https://github.com/CharlieMarcotte/Deep-HDR-with-Pytorch (ModelDeepHDR / DirectDeepHDR).

    Stacks the 3 LDR exposures with their radiance-domain versions (18 channels total)
    and regresses the merged HDR image directly with a 4-layer CNN:
    conv(18->100,7x7) -> conv(100->100,5x5) -> conv(100->50,3x3) -> conv(50->3,1x1) + sigmoid.

    As in the original repo, the sigmoid output is a mu-law tonemapped value in [0,1),
    not linear radiance directly (matches the original training against
    range_compressor(label)). The inverse mu-law expansion happens inside forward()
    itself, so every caller — training loop, validation, inference — always receives
    real linear HDR radiance, regardless of where it's called from.

    Train against this with a mu-law-domain loss (see loss_type="hdr_mulaw_l1" in
    utils_decoder.py). Pass return_compressed=True to get the sigmoid's native
    compressed-domain tensor directly, instead of round-tripping the returned linear
    radiance back through mu_law_compress — same value (mu_law_compress and
    mu_law_expand are exact inverses), but supervising the raw tensor the sigmoid
    actually produced avoids the redundant expand/compress pass.

    The only other deviation from the original is "same" padding instead of valid
    convolutions (the original crops the output by Constants.cnn_crop_size), so the
    output matches the input's spatial resolution as required by this pipeline.
    """

    REQUIRED_EXPOSURES = (-4.0, 0.0, 4.0)

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(18, 100, kernel_size=7, padding=3),
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
        self.layer4 = nn.Sequential(
            nn.Conv2d(50, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @classmethod
    def _sort_by_exposure(cls, videos, exposures):
        if torch.is_tensor(videos):
            # some callers (e.g. ValScheduler.merge_and_output) pass a single tensor with
            # exposures stacked along dim=1 (B, E, C, T, H, W) instead of a list of E tensors
            videos = [videos[:, i] for i in range(videos.shape[1])]
        assert len(videos) == 3, f"deephdr merger expects exactly 3 exposures, got {len(videos)}"

        if not torch.is_tensor(exposures):
            exposures = torch.as_tensor(exposures)
        exposures = exposures.reshape(-1).to(dtype=torch.float32)
        assert exposures.numel() == 3, f"deephdr merger expects exactly 3 exposures, got {exposures.numel()}"

        got = sorted(exposures.tolist())
        expected = sorted(cls.REQUIRED_EXPOSURES)
        assert all(abs(g - e) < 1e-3 for g, e in zip(got, expected)), \
            f"deephdr merger requires exposures {expected}, got {got}"

        order = torch.argsort(exposures).tolist()
        sorted_videos = [videos[i] for i in order]
        sorted_exposures = [exposures[i].item() for i in order]
        return sorted_videos, sorted_exposures

    def forward(self, videos, exposures, encoder_decoder_mode=None, mem_efficient=False, chunk_pixels=None, return_compressed=False):
        # videos: list of 3 tensors (B,3,T,H,W), one per exposure, in [0,1]
        # exposures: EV values for each entry of `videos` (any order)
        videos, exposures = self._sort_by_exposure(videos, exposures)
        low, normal, high = videos  # EV -4, 0, 4
        exp_low, exp_normal, exp_high = exposures

        radiance_low = low * (2.0 ** (-exp_low))
        radiance_normal = normal * (2.0 ** (-exp_normal))
        radiance_high = high * (2.0 ** (-exp_high))

        stacked = torch.cat(
            [low, normal, high, radiance_low, radiance_normal, radiance_high], dim=1
        )  # (B,18,T,H,W)

        B, _, T, H, W = stacked.shape
        x = rearrange(stacked, 'b c t h w -> (b t) c h w')
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        compressed = self.layer4(x)  # mu-law tonemapped domain, sigmoid-bounded in [0,1)
        compressed = rearrange(compressed, '(b t) c h w -> b c t h w', b=B, t=T)
        linear_hdr = mu_law_expand(compressed)  # -> real linear HDR radiance
        if return_compressed:
            return linear_hdr, compressed
        return linear_hdr
