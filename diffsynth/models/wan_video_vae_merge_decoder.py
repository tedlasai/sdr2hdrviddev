import torch
import torch.nn as nn
import sys
sys.path.append("../../")
from utils import merge_hdr, merge_hdr_avg_linear_boost  # keep if you need it
from einops import rearrange


class ExposureTokenMLP(nn.Module):
    """MLP that maps 7D token -> d_model."""
    def __init__(self, d_model=64, mlp_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(self, x):
        # x: (..., 7)
        return self.net(x)


class ExposureSelfAttention(nn.Module):
    """
    Self-attention over exposures for each pixel independently.
    Sequence length = E, batch = (B*T*H*W) (possibly chunked).
    """
    def __init__(self, d_model=64, nhead=4, attn_dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (N, E, d_model)
        # pre-norm + residual
        y = self.ln(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        return x + y


class WanVideoVAEMergeDecoder(nn.Module):
    """
    Flexible exposure/latent merge using per-pixel transformer over exposures.

    Inputs:
      - videos: list of E tensors each (B, 3, T, H, W), OR a tensor (B, E, 3, T, H, W)
      - exposures: list/1D tensor length E (EV values, same meaning as your current code)
    Output:
      - out: (B, 3, T, H, W)
    """
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        mlp_hidden: int = 128,
        num_attn_layers: int = 1,
        predict_per_channel: bool = False,  # weights shape (E,3) vs (E,1)
    ):
        super().__init__()
        self.token_mlp = ExposureTokenMLP(d_model=d_model, mlp_hidden=mlp_hidden)
        self.attn_blocks = nn.Sequential(*[
            ExposureSelfAttention(d_model=d_model, nhead=nhead)
            for _ in range(num_attn_layers)
        ])
        self.predict_per_channel = predict_per_channel

        out_dim = 3 if predict_per_channel else 1
        self.weight_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim)
        )

    @torch.no_grad()
    def _check(self, vids):
        assert vids.dim() == 6, f"expected (B,E,C,T,H,W), got {tuple(vids.shape)}"
        B, E, C, T, H, W = vids.shape
        assert C == 3, "expects RGB"
        return B, E, C, T, H, W

    def _stack_inputs(self, videos):
        if isinstance(videos, (list, tuple)):
            # list of E tensors: each (B,3,T,H,W) -> (B,E,3,T,H,W)
            vids = torch.stack(videos, dim=1)
        else:
            vids = videos
        return vids

    def forward(
        self,
        videos,
        exposures,
        encoder_decoder_mode,
        mem_efficient: bool = False,
        chunk_pixels: int = 131072,  # used when mem_efficient=True
    ) -> torch.Tensor:

        vids = self._stack_inputs(videos)  # (B,E,3,T,H,W)
        B, E, C, T, H, W = self._check(vids)

        # exposures: (E,)
        if not torch.is_tensor(exposures):
            exposures = torch.tensor(exposures, device=vids.device, dtype=vids.dtype)
        exposures = exposures.to(device=vids.device, dtype=vids.dtype).view(1, E, 1, 1, 1, 1)

        # LDR inputs
        ldr = vids  # (B,E,3,T,H,W)

        # Radiance scaling (matches your convention: radiance = ldr * 2**(-EV))
        radiance = ldr * (2.0 ** (-exposures))  # broadcast over B,C,T,H,W

        # exposure scalar channel for tokens
        exp_scalar = exposures.expand(B, E, 1, T, H, W)  # (B,E,1,T,H,W)

        # Build per-pixel exposure tokens: [ldr_rgb(3), rad_rgb(3), exp(1)] => 7
        # tokens_7: (B,E,7,T,H,W)
        tokens_7 = torch.cat([ldr, radiance, exp_scalar], dim=2)

        # Special-case: classic merge override
        if encoder_decoder_mode == "seperate_debevec":
            # keep your merge_hdr behavior (expects normal/low/high), but only valid if E==3
            assert E == 3, "seperate_debevec expects exactly 3 exposures"
            low_idx = (exposures == -4).nonzero(as_tuple=True)[1].item()
            normal_idx = (exposures == 0).nonzero(as_tuple=True)[1].item()
            high_idx = (exposures == 4).nonzero(as_tuple=True)[1].item()
            normal, low, high = vids[:, normal_idx], vids[:, low_idx], vids[:, high_idx]
            normal_r, low_r, high_r = radiance[:, normal_idx], radiance[:, low_idx], radiance[:, high_idx]

            merge_out = merge_hdr(normal, low, high, normal_r, low_r, high_r)
            return merge_out

        # Output buffer
        out = vids.new_empty((B, C, T, H, W))

        # We do attention over exposures (E) for each pixel independently.
        # Flatten pixels into N = B*T*H*W “batch” for MHA, sequence length = E.
        if not mem_efficient:
            # tokens_7: (B,E,7,T,H,W) -> (B,T,H,W,E,7) -> (N,E,7)
            x = rearrange(tokens_7, 'b e d t h w -> (b t h w) e d')

            x = self.token_mlp(x)           # (N,E,d_model)
            x = self.attn_blocks(x)         # (N,E,d_model)
            logits = self.weight_head(x)    # (N,E,3) or (N,E,1)

            if self.predict_per_channel:
                # softmax over exposures for each channel independently
                weights = torch.softmax(logits, dim=1)  # (N,E,3)
                rad = rearrange(radiance, 'b e c t h w -> (b t h w) e c')  # (N,E,3)
                fused = (weights * rad).sum(dim=1)  # (N,3)
            else:
                weights = torch.softmax(logits, dim=1)  # (N,E,1)
                rad = rearrange(radiance, 'b e c t h w -> (b t h w) e c')  # (N,E,3)
                fused = (weights * rad).sum(dim=1)  # (N,3)

            out = rearrange(fused, '(b t h w) c -> b c t h w', b=B, t=T, h=H, w=W)

        else:
            # Chunk over pixels to reduce peak memory
            N = B * T * H * W
            # Pre-flatten radiance once
            rad_all = rearrange(radiance, 'b e c t h w -> (b t h w) e c')  # (N,E,3)
            tok_all = rearrange(tokens_7, 'b e d t h w -> (b t h w) e d')  # (N,E,7)

            fused_all = vids.new_empty((N, 3))
            for start in range(0, N, chunk_pixels):
                end = min(start + chunk_pixels, N)
                x = tok_all[start:end]          # (chunk,E,7)
                x = self.token_mlp(x)           # (chunk,E,d_model)
                x = self.attn_blocks(x)         # (chunk,E,d_model)
                logits = self.weight_head(x)    # (chunk,E,3) or (chunk,E,1)

                if self.predict_per_channel:
                    w = torch.softmax(logits, dim=1)      # (chunk,E,3)
                    fused = (w * rad_all[start:end]).sum(dim=1)  # (chunk,3)
                else:
                    w = torch.softmax(logits, dim=1)      # (chunk,E,1)
                    fused = (w * rad_all[start:end]).sum(dim=1)  # (chunk,3)

                fused_all[start:end] = fused

            out = rearrange(fused_all, '(b t h w) c -> b c t h w', b=B, t=T, h=H, w=W)

        return out