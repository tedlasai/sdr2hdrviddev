"""
Benchmark WanVideoVAELatentMerger (the lediff/latent merger): forward-pass time,
peak GPU memory, and parameter count, for the latent corresponding to a single
704x1280x3 image merged from 3 exposures.

Wan2.1 VAE: z_dim=16, 8x spatial downsample -> latent H,W = 704/8, 1280/8 = 88, 160.
For a single-frame image (T=1) the causal VAE keeps T=1 in latent space too.
"""
import sys
sys.path.insert(0, "/data2/saikiran.tedla/hdrvideo/diff")

import time
import torch
from diffsynth.models.wan_video_vae_latent_merger import WanVideoVAELatentMerger

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
H, W = 704, 1280
LATENT_CHANNELS = 16
VAE_SPATIAL_DOWNSAMPLE = 8
LH, LW = H // VAE_SPATIAL_DOWNSAMPLE, W // VAE_SPATIAL_DOWNSAMPLE
B, E, T = 1, 3, 1
NUM_WARMUP = 5
NUM_ITERS = 50

model = WanVideoVAELatentMerger(latent_channels=LATENT_CHANNELS, num_exposures=E).to(DEVICE).eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,} ({total_params/1e6:.3f} M)")
print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.3f} M)")

latents = torch.randn(B, E, LATENT_CHANNELS, T, LH, LW, device=DEVICE)
exposures = torch.tensor([-4.0, 0.0, 4.0], device=DEVICE)

if DEVICE.startswith("cuda"):
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize(DEVICE)

with torch.no_grad():
    for _ in range(NUM_WARMUP):
        model(latents, exposures)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize(DEVICE)

    times = []
    for _ in range(NUM_ITERS):
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize(DEVICE)
        start = time.perf_counter()
        out = model(latents, exposures)
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize(DEVICE)
        times.append(time.perf_counter() - start)

times = torch.tensor(times)
print(f"\nInput: 3 exposures, each {B}x{E}x{LATENT_CHANNELS}x{T}x{LH}x{LW} latent on {DEVICE}")
print(f"  (corresponds to {B}x3x{T}x{H}x{W} pixel-space image at {VAE_SPATIAL_DOWNSAMPLE}x VAE downsample)")
print(f"Output shape: {tuple(out.shape)}, range: [{out.min():.3f}, {out.max():.3f}]")
print(f"\nForward pass over {NUM_ITERS} iters (after {NUM_WARMUP} warmup):")
print(f"  mean: {times.mean()*1000:.2f} ms")
print(f"  std:  {times.std()*1000:.2f} ms")
print(f"  min:  {times.min()*1000:.2f} ms")
print(f"  max:  {times.max()*1000:.2f} ms")
print(f"  FPS:  {1.0/times.mean():.2f}")

if DEVICE.startswith("cuda"):
    peak_alloc = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(DEVICE) / (1024 ** 2)
    print(f"\nPeak GPU memory allocated: {peak_alloc:.2f} MB")
    print(f"Peak GPU memory reserved:  {peak_reserved:.2f} MB")
