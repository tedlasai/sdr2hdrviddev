import torch
import torch.nn as nn
import sys
sys.path.append("../../")
from utils import merge_hdr
from .nafnet_arch import NAFNet
from einops import rearrange

import torch
import torch.nn as nn

class Simple1x1Net(nn.Module):
    def __init__(self, in_channel=9, out_channel=9, hidden_dim=64, num_layers=3):
        super(Simple1x1Net, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channel, hidden_dim, kernel_size=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(hidden_dim, out_channel, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WanVideoVAEMergeDecoder(nn.Module):
    """
    Merge three exposure stacks (normal/low/high), each (B, 3, 5, H, W),
    into a single (B, 3, 5, H, W) output using a very shallow 3D Conv net.
    """
    def __init__(self, mid_channels: int = 32, temporal_kernel: int = 1):
        super().__init__()
        assert temporal_kernel in (1, 3), "temporal_kernel must be 1 or 3"
        in_ch = 3 * 3  # normal + low + high (RGB each)
        pad_t = (temporal_kernel - 1) // 2

        # self.fuse = nn.Sequential(
        #     nn.Conv3d(in_ch, mid_channels,
        #               kernel_size=(temporal_kernel, 3, 3),
        #               padding=(pad_t, 1, 1)),
        #     nn.SiLU(),
        #     nn.Conv3d(mid_channels, 9, kernel_size=1)
        # )

        self.merge = Simple1x1Net(in_channel=9, out_channel=9, hidden_dim=16, num_layers=3) #NAFNet(in_channel=9, out_channel=9, width=16, middle_blk_num=1, enc_blk_nums=[1,1], dec_blk_nums=[1,1])

    @torch.no_grad()
    def _check(self, normal, low, high):
        assert normal.dim() == low.dim() == high.dim() == 5, "expects 5D tensors"
        Bn, Cn, Tn, Hn, Wn = normal.shape
        for e in (low, high):
            Be, Ce, Te, He, We = e.shape
            assert (Be, Ce, Te, He, We) == (Bn, Cn, Tn, Hn, Wn), "shape mismatch"
        assert Cn == 3 and Tn == 5, "expects (B,3,5,H,W)"

    def forward(self,
                normal: torch.Tensor,
                low: torch.Tensor,
                high: torch.Tensor) -> torch.Tensor:
        """
        normal/low/high: (B, 3, 5, H, W)
        returns: (B, 3, 5, H, W)
        """
        self._check(normal, low, high)
        low_radiance = low * (2 ** 4)      # EV -4
        high_radiance = high * (2 ** -4)   # EV +4
        normal_radiance = normal           # EV 0
        
                
        # normal, low, high:           (B, C, T, H, W)
        # normal_radiance, low_radiance, high_radiance: same shape

        # 1) Add exposure dim: (B, C, E, T, H, W) with E=3
        stacked = torch.stack([normal, low, high], dim=2)   # (B, C, 3, T, H, W)

        B, C, E, T, H, W = stacked.shape
        assert E == 3, f"Expected 3 exposures, got {E}"

        # 2) Run your 2D conv over (E*C) channels, merging (B,T)
        #    -> expects self.merge: (B*T, E*C, H, W) -> (B*T, E*C, H, W)
        x = rearrange(stacked, 'b c e t h w -> (b t) (e c) h w')        # (B*T, E*C, H, W)
        logits = self.merge(x)                                          # (B*T, E*C, H, W)

        # 3) Reshape back and softmax across exposures (E)
        logits = rearrange(logits, '(b t) (e c) h w -> b t e c h w', b=B, e=E, c=C)  # (B, T, E, C, H, W)
        weights = torch.sigmoid(logits)
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)  # (B, T, E, C, H, W)

        # 4) Stack radiances the same way
        radiance = torch.stack([normal_radiance, low_radiance, high_radiance], dim=2)  # (B, C, E, T, H, W)
        radiance = rearrange(radiance, 'b c e t h w -> b t e c h w')                   # (B, T, E, C, H, W)

        # 5) Fuse over exposures and return (B, C, T, H, W)
        fused = (weights * radiance).sum(dim=2)                      # (B, T, C, H, W)
        out = rearrange(fused, 'b t c h w -> b c t h w')             # (B, C, T, H, W)
        merge_out = merge_hdr(normal, low, high, normal_radiance, low_radiance, high_radiance) 
        out = merge_out + out * 0
        return out


        #classic merge
        #x = 
