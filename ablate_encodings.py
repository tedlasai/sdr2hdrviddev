"""
ablate_encodings.py

Encodes and decodes Stuttgart HDR GT frames through the Wan VAE using 4 different
encoding strategies, writing EXR outputs to:
  evaluations/ablatelinear_stuttgart/auto/{scene}/frame_XXXX.exr
  evaluations/ablatebracketed_stuttgart/auto/{scene}/frame_XXXX.exr
  evaluations/ablateloglinear_stuttgart/auto/{scene}/frame_XXXX.exr
  evaluations/ablatepu_stuttgart/auto/{scene}/frame_XXXX.exr
  evaluations/ablatebracketedmerger_stuttgart/auto/{scene}/frame_XXXX.exr
"""

import sys
import os
sys.path.insert(0, "/data2/saikiran.tedla/hdrvideo/diff")

import argparse
import json
import torch
import numpy as np
import OpenEXR
import Imath
from pathlib import Path

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_vae_merge_decoder import WanVideoVAEMergeDecoder
from diffsynth.models.utils import load_state_dict
from utils import pu21_encode_linear_video, pu21_decode_pu_video, PU21_IN_MAX_VALUE, merge_hdr

MERGE_DECODER_CKPT = "/data2/saikiran.tedla/hdrvideo/diff/models/train/finetune_decoder_mergeragain/checkpoints/epoch-4.safetensors"

GT_DIR = Path("/data2/saikiran.tedla/hdrvideo/diff/evaluations/stuttgart/hdr")
OUT_BASE = Path("/data2/saikiran.tedla/hdrvideo/diff/evaluations")
NUM_FRAMES = 17
GAMMA = 2.2


def load_exr(path):
    f = OpenEXR.InputFile(str(path))
    dw = f.header()['dataWindow']
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    PT = Imath.PixelType(Imath.PixelType.FLOAT)
    r = np.frombuffer(f.channel('R', PT), dtype=np.float32).reshape(H, W)
    g = np.frombuffer(f.channel('G', PT), dtype=np.float32).reshape(H, W)
    b = np.frombuffer(f.channel('B', PT), dtype=np.float32).reshape(H, W)
    return np.stack([r, g, b], axis=-1)  # (H, W, 3)


def save_exr(path, img_hwc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W, _ = img_hwc.shape
    img_hwc = img_hwc.astype(np.float32)
    header = OpenEXR.Header(W, H)
    ch = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {'R': ch, 'G': ch, 'B': ch}
    exr = OpenEXR.OutputFile(str(path), header)
    exr.writePixels({
        'R': img_hwc[:, :, 0].tobytes(),
        'G': img_hwc[:, :, 1].tobytes(),
        'B': img_hwc[:, :, 2].tobytes(),
    })
    exr.close()


def load_scene(scene_dir, num_frames):
    """Load first `num_frames` EXR frames -> ((1, T, C, H, W) float32 tensor, scale float)."""
    frames = []
    for i in range(num_frames):
        p = scene_dir / f"frame_{i:04d}.exr"
        if not p.exists():
            print(f"  Warning: {p} not found, stopping at frame {i}")
            break
        hwc = load_exr(p)
        frames.append(torch.from_numpy(hwc).permute(2, 0, 1))  # (C, H, W)
    video = torch.stack(frames, dim=0).unsqueeze(0)  # (1, T, C, H, W)
    scale = (0.7 / video.mean()).clamp(min=1e-8)  # auto-exposure: mean -> 0.7
    return (video * scale).clamp(max=16.0), scale.item()


def save_scene(frames_tchw, out_dir, scale):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scale.json").write_text(json.dumps({"scale": scale}))
    for i, frame in enumerate(frames_tchw):
        hwc = frame.permute(1, 2, 0).numpy()  # (H, W, C)
        hwc = np.clip(hwc, 0.0, None)
        save_exr(out_dir / f"frame_{i:04d}.exr", hwc)


def vae_encode_decode(video_01, pipe):
    """video_01: (1, T, C, H, W) in [0,1] on pipe.device -> (1, T, C, H, W) cpu float32 in [0,1]."""
    with torch.no_grad():
        v = (video_01 * 2.0 - 1.0).permute(0, 2, 1, 3, 4)  # -> (B, C, T, H, W)
        latents = pipe.vae.encode(v, device=pipe.device, tiled=False)
        latents = latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        out = pipe.vae.decode(latents, device=pipe.device, tiled=False)
        out = out.to(dtype=torch.float32, device="cpu")
        out = ((out + 1.0) / 2.0).clamp(0.0, 1.0)
        out = out.permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)
    return out


# ── encoding methods ──────────────────────────────────────────────────────────

def encode_linear(hdr, pipe):
    """hdr: (1, T, C, H, W) linear. Returns (T, C, H, W) linear."""
    max_val = hdr.max()
    encoded = (hdr / max_val).clamp(0.0, 1.0).to(device=pipe.device, dtype=pipe.torch_dtype)
    decoded = vae_encode_decode(encoded, pipe)
    return (decoded * max_val.cpu())[0]


def encode_log_linear(hdr, pipe):
    """Gamma (log) encode before VAE, undo after."""
    max_val = hdr.max()
    lin_01 = (hdr / max_val).clamp(min=1e-10)
    gamma_enc = torch.pow(lin_01, 1.0 / GAMMA).to(device=pipe.device, dtype=pipe.torch_dtype)
    decoded_01 = vae_encode_decode(gamma_enc, pipe)
    return (torch.pow(decoded_01.clamp(min=0.0), GAMMA) * max_val.cpu())[0]


def encode_pu(hdr, pipe):
    """PU21 encode before VAE, PU21 decode after."""
    scale = 1000.0 / hdr.max().item()
    pu = pu21_encode_linear_video(hdr * scale)
    pu_01 = (pu / PU21_IN_MAX_VALUE).clamp(0.0, 1.0).to(device=pipe.device, dtype=pipe.torch_dtype)
    decoded_01 = vae_encode_decode(pu_01, pipe).cpu()
    decoded_pu = decoded_01 * PU21_IN_MAX_VALUE
    return pu21_decode_pu_video(decoded_pu)[0] / scale


def _encode_brackets(hdr, pipe):
    """Shared bracket encode/decode step. Returns (max_val, dec_ev0, dec_m4, dec_p4)."""
    max_val = torch.tensor(16.0).to(device=hdr.device, dtype=hdr.dtype)
    hdr_01 = (hdr / max_val).clamp(0.0, 1.0)

    ev0 = hdr_01
    ev_m4 = (hdr_01 * (2.0 ** -4)).clamp(0.0, 1.0)  # dark bracket, preserves highlights
    ev_p4 = (hdr_01 * (2.0 ** 4)).clamp(0.0, 1.0)   # bright bracket, preserves shadows

    dec_ev0 = vae_encode_decode(ev0.to(device=pipe.device, dtype=pipe.torch_dtype), pipe).cpu()
    dec_m4  = vae_encode_decode(ev_m4.to(device=pipe.device, dtype=pipe.torch_dtype), pipe).cpu()
    dec_p4  = vae_encode_decode(ev_p4.to(device=pipe.device, dtype=pipe.torch_dtype), pipe).cpu()
    return max_val, dec_ev0, dec_m4, dec_p4


def encode_bracketed(hdr, pipe):
    """Generate EV-4/0/+4 brackets, encode/decode each, merge HDR with classic Debevec."""
    max_val, dec_ev0, dec_m4, dec_p4 = _encode_brackets(hdr, pipe)

    normal_radiance = dec_ev0
    low_radiance    = dec_m4 * (2.0 ** 4)
    high_radiance   = dec_p4 * (2.0 ** -4)

    merged = merge_hdr(dec_ev0, dec_m4, dec_p4, normal_radiance, low_radiance, high_radiance)
    merged = merged.clamp(max=1)
    return (merged * max_val.cpu())[0]


def encode_bracketed_merger(hdr, pipe, merge_decoder):
    """Generate EV-4/0/+4 brackets, encode/decode each, merge with learned WanVideoVAEMergeDecoder."""
    max_val, dec_ev0, dec_m4, dec_p4 = _encode_brackets(hdr, pipe)

    # vae_encode_decode returns (B, T, C, H, W); merge_decoder needs (B, C, T, H, W)
    dev = merge_decoder.token_mlp.net[0].weight.device
    videos = [
        dec_m4.permute(0, 2, 1, 3, 4).to(device=dev),   # EV -4
        dec_ev0.permute(0, 2, 1, 3, 4).to(device=dev),  # EV  0
        dec_p4.permute(0, 2, 1, 3, 4).to(device=dev),   # EV +4
    ]
    exposures = torch.tensor([-4.0, 0.0, 4.0])

    with torch.no_grad():
        merged = merge_decoder(videos, exposures, encoder_decoder_mode="seperate", mem_efficient=True)

    # merge_decoder output is (B, C, T, H, W); permute to (B, T, C, H, W) then take batch 0
    merged = merged.cpu().permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)
    return (merged * max_val.cpu())[0]


# ── main ──────────────────────────────────────────────────────────────────────

METHODS = {
    "linear": encode_linear,
    "loglinear": encode_log_linear,
    "pu": encode_pu,
    "bracketed": encode_bracketed,
    "bracketedmerger": None,  # populated after merge_decoder is loaded
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                        choices=list(METHODS.keys()),
                        help="Which encoding methods to run (default: all)")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    args = parser.parse_args()

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        offload_device="cpu", skip_download=True),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors",
                        offload_device="cpu", skip_download=True),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="Wan2.2_VAE.pth",
                        offload_device="cpu", skip_download=True),
        ],
    )
    pipe.load_models_to_device(["vae"])
    pipe.vae = pipe.vae.to(device=pipe.device)
    pipe.vae.eval()

    # Load merge decoder and wire up bracketedmerger method
    if "bracketedmerger" in args.methods:
        sd = load_state_dict(MERGE_DECODER_CKPT)
        sd_stripped = {k.removeprefix("merge_decoder."): v for k, v in sd.items()}
        merge_decoder = WanVideoVAEMergeDecoder()
        merge_decoder.load_state_dict(sd_stripped, strict=True)
        merge_decoder = merge_decoder.to(device=pipe.device).eval()
        METHODS["bracketedmerger"] = lambda hdr, pipe: encode_bracketed_merger(hdr, pipe, merge_decoder)
        print("Merge decoder loaded from", MERGE_DECODER_CKPT)

    scenes = sorted([d for d in GT_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(scenes)} scenes: {[s.name for s in scenes]}")

    for method_name in args.methods:
        fn = METHODS[method_name]
        out_dir = OUT_BASE / f"ablate{method_name}_stuttgart" / "auto"
        print(f"\n=== Method: {method_name} -> {out_dir} ===")

        for scene_dir in scenes:
            scene_name = scene_dir.name
            scene_out = out_dir / scene_name
            print(f"  {scene_name} ...", end=" ", flush=True)

            hdr, scene_scale = load_scene(scene_dir, args.num_frames)
            result = fn(hdr, pipe)  # (T, C, H, W)
            save_scene(result, scene_out, scene_scale)
            print(f"done ({result.shape[0]} frames)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
