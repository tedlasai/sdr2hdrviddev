"""
Quick test: load WanVideoVAEMergeDecoder from checkpoint and run a forward pass.
"""
import sys
sys.path.insert(0, "/data2/saikiran.tedla/hdrvideo/diff")

import torch
from diffsynth.models.wan_video_vae_merge_decoder import WanVideoVAEMergeDecoder
from diffsynth.models.utils import load_state_dict

CKPT = "/data2/saikiran.tedla/hdrvideo/diff/models/train/finetune_decoder_mergeragain/checkpoints/epoch-4.safetensors"
DEVICE = "cuda:7"

# ── 1. Load checkpoint ────────────────────────────────────────────────────────
sd = load_state_dict(CKPT)
print("Checkpoint keys:", list(sd.keys()))

# Strip the 'merge_decoder.' prefix saved by the training module
sd_stripped = {k.removeprefix("merge_decoder."): v for k, v in sd.items()}

model = WanVideoVAEMergeDecoder()
missing, unexpected = model.load_state_dict(sd_stripped, strict=True)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")
model = model.to(DEVICE).eval()
print("Model loaded OK!")

# ── 2. Dummy forward pass ─────────────────────────────────────────────────────
B, C, T, H, W = 1, 3, 5, 64, 64
# Three brackets: EV -4, EV 0, EV +4 — each (B, C, T, H, W) in [0, 1]
videos = [torch.rand(B, C, T, H, W, device=DEVICE) for _ in range(3)]
exposures = torch.tensor([-4.0, 0.0, 4.0])

with torch.no_grad():
    out = model(videos, exposures, encoder_decoder_mode="seperate", mem_efficient=False)

print(f"Forward pass OK! Output shape: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]")
