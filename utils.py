import cv2
import numpy as np
import os
from torchmetrics.image import PeakSignalNoiseRatio
import torch
from einops import rearrange
import imageio.v3 as iio
from pathlib import Path
def output_frames(frames, out_folder, mode, channel_order='NHWC'):
    #force mode to be either 'hdr' or 'ldr'
    assert mode in ('hdr', 'ldr'), "mode must be either 'hdr' or 'ldr'"
    os.makedirs(out_folder, exist_ok=True)
    if channel_order == 'NCHW':
        frames = frames.permute(0, 2, 3, 1)  # N,C,H,W -> N,H,W,C
    N, H, W, C = frames.shape
    for i in range(N):
        frame = frames[i].cpu().numpy()  # H, W, C
        frame = frame[:, :, ::-1]  # Convert RGB to BGR
        if mode == "hdr":
            filename = f"{out_folder}/frame_{i:04d}.exr"
        else:
            filename = f"{out_folder}/frame_{i:04d}.png"
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(filename, frame)  # Save as .hdr image

import imageio

def output_ldr_video(ldr_video, out_path, channel_order='NHWC', fps=24, quality=10, ffmpeg_params=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if channel_order == 'NCHW':
        ldr_video = ldr_video.permute(0, 2, 3, 1)
    N, H, W, C = ldr_video.shape

    writer = imageio.get_writer(out_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params or [])
    for i in range(N):
        frame = ldr_video[i].cpu().numpy()
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()

from torchmetrics.image import PeakSignalNoiseRatio
def get_psnr_fn(device):
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=(1, 2, 3)).to(device=device)
    return psnr_metric

from torchmetrics.image import StructuralSimilarityIndexMeasure
def get_ssim_fn(device):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device=device)
    return ssim_metric
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
def get_lpips_fn(device):
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='none').to(device=device)
    return lpips_metric
def vid_lpips(pred, tgt, lpips_loss):
    return lpips_loss(
        rearrange(pred, 'b c f h w -> (b f) c h w') * 2 - 1,
        rearrange(tgt,  'b c f h w -> (b f) c h w') * 2 - 1
    ).mean()
def compute_all_metrics(gt, pred, tag, metrics_dict, out_metrics):
    # don't shadow metrics_dict here!
    for name in metrics_dict[tag].keys():
        fn = metrics_dict[tag][name]
        val = fn(gt, pred)                           # could be differentiable
        if isinstance(out_metrics[tag][name], list):
            out_metrics[tag][name].append(val)
        else:
            out_metrics[tag][name] = torch.cat(
                [out_metrics[tag][name], val.unsqueeze(0)], dim=0
            )

def average_metrics(out_metrics):
    """
    Averages metrics per tag and per metric.
    Args:
        out_metrics: dict[tag][metric] -> torch.Tensor (B*T,)
    Returns:
        avg_dict: dict[tag][metric] -> scalar float
    """
    avg_dict = {}
    for tag, metrics in out_metrics.items():
        avg_dict[tag] = {}
        for name, values in metrics.items():
            if isinstance(values, list):
                values = torch.stack(values, dim=0)
            avg_dict[tag][name] = values.mean().item()
    return avg_dict

def flatten_dict(d, parent_key="", sep="_"):
    """
    Recursively flattens a nested dictionary.
    Example:
        {"low": {"psnr": 31, "ssim": 0.9}} 
    ->  {"low_psnr": 31, "low_ssim": 0.9}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def generate_multi_exposure_video(hdr_video: torch.Tensor, exposures=(-4, 0, 4)) -> torch.Tensor:
    """
    Args:
        hdr_video: Tensor of shape (B, T, C, H, W), float, HDR or LDR in [0, 1+] range.
        exposures: Iterable of EV stops. Each exposure scales by 2**EV.

    Returns:
        Tensor of shape (B, E, T, C, H, W) with values clamped to [0, 1].
    """
    if hdr_video.ndim != 5:
        raise ValueError(f"Expected (B, T, C, H, W), got {hdr_video.shape}")

    B, T, C, H, W = hdr_video.shape
    E = len(exposures)

    factors = torch.tensor(exposures, dtype=hdr_video.dtype, device=hdr_video.device)
    factors = (2.0 ** factors).view(1, E, 1, 1, 1, 1)  # (1, E, 1, 1, 1, 1)

    # Add exposure axis and scale
    multi = hdr_video.unsqueeze(1) * factors  # (B, E, T, C, H, W)

    # Clip to displayable range
    return multi.clamp(0.0, 1.0)



def write_exposure_bracket(hdr_path, evs, out_dir_name="bracket"):
    hdr_path = Path(hdr_path)
    img = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED).astype(np.float32)[:,:,::-1]  # BGR to RGB

    out_dir = hdr_path.parent / out_dir_name
    out_dir.mkdir(exist_ok=True)

    for ev in evs:
        exposed = np.clip(img * (2.0 ** ev), 0.0, 1.0)
        png = (exposed * 255).astype(np.uint8)
        iio.imwrite(out_dir / f"{hdr_path.stem}_ev{ev:+.2f}.png", png)

    return out_dir

def average_frame_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Computes the average per-frame PSNR between two HDR videos.

    Args:
        pred (torch.Tensor): Predicted video of shape [B, C, T, H, W].
        target (torch.Tensor): Ground-truth video of shape [B, C, T, H, W].
        data_range (float): Max value of the signal (1.0 if normalized).

    Returns:
        torch.Tensor: Scalar tensor with average PSNR over frames.
    """
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    B, C, T, H, W = pred.shape
    #move both tensors to cpu
    pred = pred.cpu()
    target = target.cpu()
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range)

    #clamp values to [0, data_range]
    pred = torch.clamp(pred, 0.0, data_range)
    target = torch.clamp(target, 0.0, data_range)
    
    psnrs = []
    for t in range(T):
        # Extract frame t: shape [B, C, H, W]
        psnr_val = psnr_metric(pred[:, :, t], target[:, :, t])
        psnrs.append(psnr_val)

    return torch.stack(psnrs).mean()


def weight_function(video: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Triangular hat weight function.
    Args:
        video: torch.Tensor, float in [0,1] (or roughly normalized exposures)
               arbitrary shape, e.g. (B,T,C,H,W).
        eps:   small constant to avoid zero weight.
    Returns:
        weights: same shape as video, ≥ eps.
    """
    w = 1.0 - 2.0 * (video - 0.5).abs()
    w = torch.clamp(w, min=0.0)
    return w + eps

import torch
import torch.nn.functional as F

import torch

def merge_hdr_avg_linear_boost(
    normal_exposure, low_exposure, high_exposure,
    normal_radiance, low_radiance, high_radiance,
    t_low=0.05, t_high=0.95, delta=0.05, k_sat=0.5, k_black=0.5, eps=1e-8
):
    # base hat weights (channel-wise to avoid adds of None)
    w_low  = weight_function(low_exposure)
    w_norm = weight_function(normal_exposure)
    w_high = weight_function(high_exposure)

    # linear ramps (clamped 0..1)
    ramp_sat_low   = torch.clamp((low_exposure  - t_high) / max(delta, 1e-6), 0.0, 1.0)  # bright → 1
    ramp_black_high= torch.clamp((t_low - high_exposure) / max(delta, 1e-6), 0.0, 1.0)   # dark  → 1

    # === the two bias lines ===
    w_low  = w_low + (k_sat   * ramp_sat_low)
    w_high = w_high + (k_black * ramp_black_high)

    # normalized average (still linear in radiances)
    num = w_low * low_radiance + w_norm * normal_radiance + w_high * high_radiance
    den = w_low + w_norm + w_high + eps
    hdr = num / den

    return hdr



def merge_hdr(normal_exposure, low_exposure, high_exposure, normal_radiance, low_radiance, high_radiance):
    """
    Merge low, normal, and high exposure images into an HDR image.

    Args:
        low_exposure: np.ndarray of shape (N, H, W, 3), low exposure images.
        normal_exposure: np.ndarray of shape (N, H, W, 3), normal exposure images.
        high_exposure: np.ndarray of shape (N, H, W, 3), high exposure images.
    Returns:
        np.ndarray of shape (N, H, W, 3)
    """
    w_normal = weight_function(normal_exposure)
    w_low = weight_function(low_exposure)
    w_high = weight_function(high_exposure)

    #if pixel is clipped in low_exposure, set the weight to w_low very high
    w_low[low_exposure >= 0.95] = 100
    w_high[high_exposure <= 0.05] = 100

    numerator = (w_low * low_radiance) + (w_normal * normal_radiance) + (w_high * high_radiance)
    denominator = w_low + w_normal + w_high + 1e-8  # Avoid division by zero

    hdr_image = numerator / denominator
    return hdr_image

def process_bracketed_video(video):
    #assert that video has 12 frames
    assert video.shape[0] == 15, "Video must have 15 frames for bracketed HDR processing"
    normal_exposure = video[0:5] #EV 0
    low_exposure = video[5:10] #EV -4
    high_exposure = video[10:15] #EV +4

    normal_radiance = normal_exposure
    low_radiance = low_exposure * (2 ** 4)  # EV -4
    high_radiance = high_exposure * (2 ** -4)  # EV +4

    hdr_video = merge_hdr(normal_exposure,low_exposure,high_exposure, normal_radiance, low_radiance,high_radiance)

    return hdr_video

def output_hdr_video(hdr_video, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    #write out each frame of hdr_video to path as .rad file
    N, H, W, C = hdr_video.shape
    for i in range(N):
        frame = hdr_video[i].cpu().numpy()  # H, W, C
        frame = frame[:, :, ::-1]  # Convert RGB to BGR
        filename = f"{out_folder}/frame_{i:04d}.hdr"
        cv2.imwrite(filename, frame)  # Save as .hdr image


def write_exposure_bracket(hdr_path, evs, out_dir_name="bracket"):
    hdr_path = Path(hdr_path)
    img = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED).astype(np.float32)[:,:,::-1]  # BGR to RGB

    out_dir = hdr_path.parent / out_dir_name
    out_dir.mkdir(exist_ok=True)

    for ev in evs:
        exposed = np.clip(img * (2.0 ** ev), 0.0, 1.0)
        png = (exposed * 255).astype(np.uint8)
        iio.imwrite(out_dir / f"{hdr_path.stem}_ev{ev:+.2f}.png", png)

    return out_dir

def write_mp4_avc1(frames: np.ndarray, output_file: str, fps: int = 30):
    """
    Write a NumPy array of frames to an MP4 file using avc1 (H.264).

    Parameters:
        frames (np.ndarray): Array of shape (N, H, W, 3) with dtype=uint8.
                             N = number of frames, H = height, W = width, 3 = RGB channels.
        output_file (str): Output mp4 file path.
        fps (int): Frames per second.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Frames must have shape (N, H, W, 3) with 3 color channels (RGB).")

    num_frames, height, width, _ = frames.shape

    # Define video writer with avc1 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for i in range(num_frames):
        # OpenCV expects BGR, so convert from RGB
        frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved as {output_file} with {num_frames} frames at {fps} fps.")


# --- PU21 (perceptually uniform) helpers; encoder from metrics/pu21.py (Mantiuk & Azimi) ---
import sys as _sys

_metrics_dir = Path(__file__).resolve().parent / "metrics"
if str(_metrics_dir) not in _sys.path:
    _sys.path.insert(0, str(_metrics_dir))
from pu21 import PU21Encoder as _PU21Encoder  # noqa: E402

# Upper bound for scaling PU values into [0, 1] before the VAE (matches typical PU21 range).
PU21_IN_MAX_VALUE = 600.0
_pu21_default = _PU21Encoder("banding_glare")


def pu21_encode_linear_video(video: torch.Tensor) -> torch.Tensor:
    """Linear RGB (B, T, 3, H, W) in absolute/nits-like units -> PU21-encoded, same layout."""
    device, dtype = video.device, video.dtype
    arr = video.detach().float().cpu().numpy()
    b, t, c, h, w = arr.shape
    if c != 3:
        raise ValueError("PU21 path expects 3 RGB channels")
    out = np.empty((b, t, c, h, w), dtype=np.float32)
    for bi in range(b):
        for ti in range(t):
            hwc = np.transpose(arr[bi, ti], (1, 2, 0))
            pu = _pu21_default.encode(hwc).astype(np.float32)
            out[bi, ti] = np.transpose(pu, (2, 0, 1))
    return torch.from_numpy(out).to(device=device, dtype=dtype)


def pu21_decode_pu_video(video: torch.Tensor) -> torch.Tensor:
    """PU21-encoded RGB (B, T, 3, H, W) -> linear RGB, same layout."""
    device, dtype = video.device, video.dtype
    arr = video.detach().float().cpu().numpy()
    b, t, c, h, w = arr.shape
    if c != 3:
        raise ValueError("PU21 path expects 3 RGB channels")
    out = np.empty((b, t, c, h, w), dtype=np.float32)
    for bi in range(b):
        for ti in range(t):
            hwc = np.transpose(arr[bi, ti], (1, 2, 0))
            lin = _pu21_default.decode(hwc.astype(np.float64)).astype(np.float32)
            out[bi, ti] = np.transpose(lin, (2, 0, 1))
    return torch.from_numpy(out).to(device=device, dtype=dtype)