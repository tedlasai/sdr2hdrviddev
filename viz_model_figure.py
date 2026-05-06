"""
viz_model_figure.py - Saves PNGs for all pipeline stages to figures/model_viz/.

Run like test.py:
  PYTHONPATH=. accelerate launch viz_model_figure.py \\
      --config diffsynth/configs/threeexposures_crffixed_test_val.yaml

Outputs under figures/model_viz/:
  input_crf/                     - input CRF video frames
  multiexposure/ev{ev}/          - model output per-exposure LDR frames
  merging/ldr/ev{ev}/            - input LDR to merge decoder (per exposure)
  merging/radiance/ev{ev}/       - linear radiance estimates (per exposure)
  ev_maps/ev{ev}/                 - solid-grey scale reference (ev-4→black, ev0→mid-grey, ev+4→white)
  merging/ev_weights/ev{ev}/     - learned attention weight maps, turbo heatmap (per exposure)
  merging/output_radiance/       - final merged HDR visualized as radiance
  output_hdr/raw/                - HDR frames, gamma-compressed PNG
  output_hdr/tonemapped/         - HDR frames, Drago tonemapped (gamma=2.2, bias=0.85)
"""

import os
import sys
sys.path.append("/data2/saikiran.tedla/hdrvideo/diff")
sys.path.append("/data2/saikiran.tedla/hdrvideo/diff/examples/wanvideo/model_training")

import numpy as np
import torch
import cv2
from einops import rearrange
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from diffsynth.trainers.utils import wan_parser
from diffsynth.trainers.video_dataset import VideoDataset
from examples.wanvideo.model_training.train import (
    WanTrainingModule, load_yaml_config, set_load_paths,
)

OUT_BASE = "figures/model_viz"
GENERATE_EXPOSURES = (-4, 0, 4)


# --------------------------------------------------------------------------- #
# PNG writers
# --------------------------------------------------------------------------- #

def _to_np_thwc(t):
    """(C, T, H, W) or (T, H, W, C) or (T, H, W) torch/numpy → numpy THWC or THW."""
    if torch.is_tensor(t):
        t = t.detach().cpu().float().numpy()
    return t


def save_ldr_frames(video, out_dir):
    """video: (T, H, W, C) or (C, T, H, W) float [0,1] → uint8 PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    v = _to_np_thwc(video)
    if v.ndim == 4 and v.shape[0] <= 4 and v.shape[3] > 4:
        # CTHW → THWC
        v = np.transpose(v, (1, 2, 3, 0))
    for i, f in enumerate(v):
        bgr = (np.clip(f, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), bgr)


def save_radiance_frames(video, out_dir, scale=16.0):
    """video: (T, H, W, C) or (C, T, H, W) float linear radiance → PNG.
    scale: the absolute max radiance value (ldr=1 at ev-4 gives 1*2^4=16)."""
    os.makedirs(out_dir, exist_ok=True)
    v = _to_np_thwc(video)
    if v.ndim == 4 and v.shape[0] <= 4 and v.shape[3] > 4:
        v = np.transpose(v, (1, 2, 3, 0))
    v_norm = np.clip(v / scale, 0, 1) ** (1.0 / 2.2)
    for i, f in enumerate(v_norm):
        bgr = (f * 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), bgr)


def save_weight_frames(weights, out_dir):
    """weights: (T, H, W) float [0,1] → turbo heatmap PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    w = _to_np_thwc(weights)
    for i, f in enumerate(w):
        hm = cv2.applyColorMap((np.clip(f, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), hm)


def save_turbo_colorbar(out_path, height=512, width=64):
    """Saves a standalone vertical turbo colorbar PNG (1→0 top to bottom)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gradient = np.tile(np.linspace(255, 0, height, dtype=np.uint8)[:, None], (1, width))
    colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    cv2.imwrite(out_path, colorbar)


def save_ev_map_frames(ev_value, num_frames, height, width, out_dir):
    """Saves solid-grey frames showing relative EV scale.
    ev_value: EV stop (e.g. -4, 0, 4).  Maps linearly: -4→0, 0→0.5, 4→1."""
    os.makedirs(out_dir, exist_ok=True)
    brightness = np.clip((ev_value + 4) / 8.0, 0.0, 1.0)
    fill = int(round(brightness * 255))
    frame = np.full((height, width, 3), fill, dtype=np.uint8)
    for i in range(num_frames):
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), frame)


def save_hdr_raw(hdr, out_dir):
    """hdr: (C, T, H, W) float linear → 99th-pct normalised + gamma-2.2 PNG."""
    os.makedirs(out_dir, exist_ok=True)
    v = _to_np_thwc(hdr)                      # still CTHW at this point
    v = np.transpose(v, (1, 2, 3, 0))         # → THWC
    maxval = np.percentile(v, 99.9) + 1e-8
    for i, f in enumerate(v):
        f_norm = np.clip(f / maxval, 0, 1) ** (1.0 / 2.2)
        bgr = (f_norm * 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), bgr)


def save_hdr_drago(hdr, out_dir, gamma=2.2, saturation=1.0, bias=0.85):
    """hdr: (C, T, H, W) float linear → Drago-tonemapped PNG."""
    os.makedirs(out_dir, exist_ok=True)
    v = _to_np_thwc(hdr)
    v = np.transpose(v, (1, 2, 3, 0))         # THWC
    tonemap = cv2.createTonemapDrago(gamma=gamma, saturation=saturation, bias=bias)
    for i, f in enumerate(v):
        bgr_in = f[:, :, ::-1].astype(np.float32)   # RGB → BGR
        ldr = tonemap.process(bgr_in)
        ldr = np.clip(ldr, 0, 1)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), (ldr * 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Merge decoder intermediate extraction
# --------------------------------------------------------------------------- #

def extract_merge_intermediates(per_exp, sorted_evs, merge_decoder):
    """
    Re-runs the merge decoder internals on the 3 per-exposure LDR videos
    and returns per-exposure LDR, radiance, attention weights, and the output.

    per_exp: dict {ev: tensor (1, 3, T, H, W)} float [0,1]
    sorted_evs: sorted list of EV values
    merge_decoder: WanVideoVAEMergeDecoder

    Returns dicts keyed by EV:
      ldr_dict      {ev: (3, T, H, W)}
      radiance_dict {ev: (3, T, H, W)}
      weights_dict  {ev: (T, H, W)}   attention weight averaged over channels
    """
    dec = merge_decoder
    # Put everything on the same device as the decoder
    try:
        dec_device = next(dec.parameters()).device
    except StopIteration:
        dec_device = torch.device("cpu")

    vids = torch.stack([per_exp[e] for e in sorted_evs], dim=1).to(dec_device)  # (1,E,3,T,H,W)
    B, E, C, T, H, W = vids.shape

    evs_t = torch.tensor(sorted_evs, device=dec_device, dtype=vids.dtype).view(1, E, 1, 1, 1, 1)
    ldr = vids
    radiance = ldr * (2.0 ** (-evs_t))                          # (1,E,3,T,H,W)
    exp_scalar = evs_t.expand(B, E, 1, T, H, W)
    tokens_7 = torch.cat([ldr, radiance, exp_scalar], dim=2)    # (1,E,7,T,H,W)

    N = B * T * H * W
    tok_all = rearrange(tokens_7, 'b e d t h w -> (b t h w) e d')
    rad_all = rearrange(radiance, 'b e c t h w -> (b t h w) e c')

    CHUNK = 131072
    weights_all = vids.new_empty((N, E))
    fused_all = vids.new_empty((N, 3))

    with torch.no_grad():
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            x = tok_all[start:end]
            x = dec.token_mlp(x)
            x = dec.attn_blocks(x)
            logits = dec.weight_head(x)              # (chunk, E, 1 or 3)
            w = torch.softmax(logits, dim=1)         # (chunk, E, 1 or 3)
            weights_all[start:end] = w.mean(dim=-1)  # (chunk, E) — avg over colour channels
            fused_all[start:end] = (w * rad_all[start:end]).sum(dim=1)

    # Reshape
    weights_bxethw = rearrange(weights_all, '(b t h w) e -> b e t h w', b=B, t=T, h=H, w=W)
    out_bxcthw = rearrange(fused_all, '(b t h w) c -> b c t h w', b=B, t=T, h=H, w=W)

    ldr_dict = {e: ldr[0, i].cpu() for i, e in enumerate(sorted_evs)}
    radiance_dict = {e: radiance[0, i].cpu() for i, e in enumerate(sorted_evs)}
    weights_dict = {e: weights_bxethw[0, i].cpu() for i, e in enumerate(sorted_evs)}

    return ldr_dict, radiance_dict, weights_dict, out_bxcthw[0].cpu()


# --------------------------------------------------------------------------- #
# Main visualisation
# --------------------------------------------------------------------------- #

def visualize(data, model, args):
    condition_video = data["input_video"]   # numpy (T, H, W, 3) float [0, 255]

    # ── 1. Input CRF frames ─────────────────────────────────────────────────
    out_crf = os.path.join(OUT_BASE, "input_crf")
    os.makedirs(out_crf, exist_ok=True)
    for i, f in enumerate(condition_video):
        bgr = np.clip(f, 0, 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(out_crf, f"frame_{i:04d}.png"), bgr)
    print(f"[viz] input_crf: {len(condition_video)} frames → {out_crf}")

    # ── 2. Run inference ─────────────────────────────────────────────────────
    outputs = model.pipe(
        prompt=data["prompt"],
        condition_video=condition_video,
        height=args.height,
        width=args.width,
        num_inference_steps=50,
        seed=1,
        tiled=False,
        cfg_scale=1.0,
        encoder_decoder_mode=model.encoder_decoder_mode,
        exposures=data["exposures"],
        generate_exposures=GENERATE_EXPOSURES,
        use_vae_ea=getattr(model, "use_vae_ea", False),
    )

    hdr_video = outputs["hdr_video"]       # (1, 3, T, H, W) linear HDR
    combined = outputs["combined_video"]   # (1, 3, 3*T_per, H, W) LDR [0,1]

    T_hdr = hdr_video.shape[2]
    T_per = combined.shape[2] // 3        # frames per exposure (may include padding)

    # Per-exposure LDR videos: trim padding to match HDR length
    per_exp = {
        0:  combined[:, :, 0*T_per:1*T_per][:, :, :T_hdr],
        -4: combined[:, :, 1*T_per:2*T_per][:, :, :T_hdr],
        4:  combined[:, :, 2*T_per:3*T_per][:, :, :T_hdr],
    }

    # ── 3. Multiexposure output frames ───────────────────────────────────────
    for ev in GENERATE_EXPOSURES:
        ev_str = f"ev{ev:+d}"
        # per_exp[ev]: (1, 3, T, H, W) → permute to (T, H, W, 3)
        save_ldr_frames(
            per_exp[ev][0].permute(1, 2, 3, 0),
            os.path.join(OUT_BASE, "multiexposure", ev_str),
        )
    print(f"[viz] multiexposure: EVs {GENERATE_EXPOSURES}")

    # ── 3b. EV maps (solid-colour scale reference) ───────────────────────────
    H, W = hdr_video.shape[3], hdr_video.shape[4]
    for ev in GENERATE_EXPOSURES:
        save_ev_map_frames(
            ev, T_hdr, H, W,
            os.path.join(OUT_BASE, "ev_maps", f"ev{ev:+d}"),
        )
    print(f"[viz] ev_maps: EVs {GENERATE_EXPOSURES} (ev-4→black, ev0→mid-grey, ev+4→white)")

    # ── 4. Merge decoder intermediates ───────────────────────────────────────
    sorted_evs = sorted(GENERATE_EXPOSURES)
    ldr_dict, radiance_dict, weights_dict, out_radiance = extract_merge_intermediates(
        per_exp, sorted_evs, model.pipe.merge_decoder
    )

    for ev in sorted_evs:
        ev_str = f"ev{ev:+d}"
        save_ldr_frames(
            ldr_dict[ev].permute(1, 2, 3, 0),
            os.path.join(OUT_BASE, "merging", "ldr", ev_str),
        )
        save_radiance_frames(
            radiance_dict[ev].permute(1, 2, 3, 0),
            os.path.join(OUT_BASE, "merging", "radiance", ev_str),
        )
        save_weight_frames(
            weights_dict[ev],
            os.path.join(OUT_BASE, "merging", "ev_weights", ev_str),
        )
    save_turbo_colorbar(os.path.join(OUT_BASE, "merging", "ev_weights_colorbar.png"))
    print("[viz] merging intermediates: ldr / radiance / ev_weights")

    save_radiance_frames(
        hdr_video[0].permute(1, 2, 3, 0),
        os.path.join(OUT_BASE, "merging", "output_radiance"),
    )
    print("[viz] merging/output_radiance")

    # ── 5. Output HDR ────────────────────────────────────────────────────────
    save_hdr_raw(hdr_video[0], os.path.join(OUT_BASE, "output_hdr", "raw"))
    save_hdr_drago(hdr_video[0], os.path.join(OUT_BASE, "output_hdr", "tonemapped"))
    print("[viz] output_hdr: raw + drago tonemapped")

    print(f"\n[viz] All outputs in {OUT_BASE}/")

    # ── 6. Square-crop all PNGs to 704×704 (centre crop, in-place) ───────────
    crop_size = 704
    count = 0
    for root, _dirs, files in os.walk(OUT_BASE):
        for fname in files:
            if not fname.lower().endswith(".png"):
                continue
            path = os.path.join(root, fname)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            h, w = img.shape[:2]
            top  = (h - crop_size) // 2
            left = (w - crop_size) // 2
            if top < 0 or left < 0:
                print(f"[viz] WARNING: {path} is smaller than {crop_size}px, skipping crop")
                continue
            img = img[top:top + crop_size, left:left + crop_size]
            cv2.imwrite(path, img)
            count += 1
    print(f"[viz] Centre-cropped {count} PNGs to {crop_size}×{crop_size}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    args = load_yaml_config(args, args.config)
    args = set_load_paths(args)

    val_dataset = VideoDataset(
        base_path="/data2/saikiran.tedla/hdrvideo/diff/evaluations/stuttgart/over/carousel_fireworks_02",
        out_path="/data2/saikiran.tedla/hdrvideo/diff/evaluations/viz_output",
        main_data_operator=VideoDataset.default_video_operator(
            num_frames=17,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
        ),
    )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        encoder_decoder_mode=args.encode_decoder_mode,
        use_vae_ea=getattr(args, "use_vae_ea", False),
    )

    accelerator = Accelerator(
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=4
    )
    model, val_dataloader = accelerator.prepare(model, val_dataloader)

    data = next(iter(val_dataloader))
    unwrapped = accelerator.unwrap_model(model)

    with torch.no_grad():
        visualize(data, unwrapped, args)
