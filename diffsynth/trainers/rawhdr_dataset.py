"""
RawHDR dataset utilities: convert .mat (4-ch RGBG + wb + cam2rgb) to linear 3-ch EXR.

Output layout (next to RawHDRTrain / RawHDRTest):
  RawHDR/RawHDRTrain_EXR/*.exr
  RawHDR/RawHDRTest_EXR/*.exr

Run:
  python -m diffsynth.trainers.rawhdr_dataset
  python -m diffsynth.trainers.rawhdr_dataset --rawhdr-root /path/to/RawHDR
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import torch
from tqdm import tqdm

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

DEFAULT_RAWHDR_ROOT = "/data2/saikiran.tedla/hdrvideo/diff/data/RawHDR"
MAT_SPLITS = ("RawHDRTrain", "RawHDRTest")
DEFAULT_DOWNSAMPLE = 2  # 2x2 box-filter average per axis


def box_downsample(rgb: np.ndarray, factor: int = 2) -> np.ndarray:
    """Average pooling with a factor x factor box filter (e.g. 2x2)."""
    if factor <= 1:
        return rgb
    h, w = rgb.shape[:2]
    h2, w2 = h - h % factor, w - w % factor
    x = rgb[:h2, :w2]
    if rgb.ndim == 2:
        return x.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))
    return x.reshape(h2 // factor, factor, w2 // factor, factor, rgb.shape[2]).mean(axis=(1, 3))


def rawhdr_packed_to_linear_rgb(
    gt: np.ndarray,
    wb: np.ndarray,
    cam2rgb: np.ndarray,
    data_range: float = 8.0,
) -> np.ndarray:
    """
    RawHDR gt is 4-ch RGBG (C, H, W). Apply WB, bin to RGB, then cam2rgb CCM.
    Returns linear RGB float32 (H, W, 3), no gamma.
    """
    x = torch.from_numpy(np.asarray(gt, dtype=np.float32)).unsqueeze(0)  # 1,4,H,W
    wb = torch.from_numpy(np.asarray(wb, dtype=np.float32)).reshape(-1)
    cam2rgb = torch.from_numpy(np.asarray(cam2rgb, dtype=np.float32))

    if wb.numel() == 4:
        x = x * wb.view(1, 4, 1, 1)
    elif wb.numel() == 3:
        x = x[:, :3] * wb.view(1, 3, 1, 1)
    else:
        raise ValueError(f"Unexpected wb shape: {wb.shape}")

    x = torch.clamp(x, 0.0, data_range)
    rgb = torch.stack(
        [x[0, 0], (x[0, 1] + x[0, 3]) * 0.5, x[0, 2]],
        dim=0,
    )  # 3,H,W
    rgb = rgb.permute(1, 2, 0)  # H,W,3
    rgb = torch.matmul(rgb, cam2rgb.T)
    return rgb.numpy().astype(np.float32)


def load_rawhdr_gt_from_mat(
    mat_path: str | Path,
    data_range: float = 8.0,
    downsample: int = DEFAULT_DOWNSAMPLE,
) -> np.ndarray:
    data = scipy.io.loadmat(str(mat_path), variable_names=["gt", "wb", "cam2rgb"])
    rgb = rawhdr_packed_to_linear_rgb(data["gt"], data["wb"], data["cam2rgb"], data_range=data_range)
    return box_downsample(rgb, factor=downsample)


def convert_mat_to_exr(
    mat_path: str | Path,
    exr_path: str | Path,
    data_range: float = 8.0,
    downsample: int = DEFAULT_DOWNSAMPLE,
    overwrite: bool = False,
) -> bool:
    mat_path = Path(mat_path)
    exr_path = Path(exr_path)
    if exr_path.exists() and not overwrite:
        return False

    rgb = load_rawhdr_gt_from_mat(mat_path, data_range=data_range, downsample=downsample)
    exr_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    if not cv2.imwrite(str(exr_path), bgr):
        raise RuntimeError(f"Failed to write EXR: {exr_path}")
    return True


def convert_split(
    rawhdr_root: str | Path,
    split: str,
    data_range: float = 8.0,
    downsample: int = DEFAULT_DOWNSAMPLE,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """
    Convert all .mat in {split}/ to {split}_EXR/*.exr.
    Returns (written, skipped, failed).
    """
    rawhdr_root = Path(rawhdr_root)
    mat_dir = rawhdr_root / split
    exr_dir = rawhdr_root / f"{split}_EXR"
    if not mat_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {mat_dir}")

    mat_files = sorted(mat_dir.glob("*.mat"))
    written = skipped = failed = 0
    for mat_path in tqdm(mat_files, desc=f"{split} -> {split}_EXR"):
        exr_path = exr_dir / f"{mat_path.stem}.exr"
        try:
            if convert_mat_to_exr(
                mat_path, exr_path, data_range=data_range, downsample=downsample, overwrite=overwrite
            ):
                written += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"ERROR {mat_path.name}: {exc}")
    return written, skipped, failed


def convert_all(
    rawhdr_root: str | Path = DEFAULT_RAWHDR_ROOT,
    splits: tuple[str, ...] = MAT_SPLITS,
    data_range: float = 8.0,
    downsample: int = DEFAULT_DOWNSAMPLE,
    overwrite: bool = False,
) -> None:
    rawhdr_root = Path(rawhdr_root)
    totals = {"written": 0, "skipped": 0, "failed": 0}
    for split in splits:
        w, s, f = convert_split(
            rawhdr_root, split, data_range=data_range, downsample=downsample, overwrite=overwrite
        )
        print(f"{split}_EXR: wrote {w}, skipped {s}, failed {f}")
        totals["written"] += w
        totals["skipped"] += s
        totals["failed"] += f
    print(
        f"Done. wrote={totals['written']} skipped={totals['skipped']} failed={totals['failed']} "
        f"under {rawhdr_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RawHDR .mat gt to linear 3-ch EXR.")
    parser.add_argument(
        "--rawhdr-root",
        type=str,
        default=DEFAULT_RAWHDR_ROOT,
        help="Root containing RawHDRTrain/ and RawHDRTest/",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=list(MAT_SPLITS),
        help="Which splits to convert (default: RawHDRTrain RawHDRTest)",
    )
    parser.add_argument("--data-range", type=float, default=8.0, help="Clamp before CCM (RawHDR default 8)")
    parser.add_argument(
        "--downsample",
        type=int,
        default=DEFAULT_DOWNSAMPLE,
        help="Box-filter downsample factor after RGB conversion (default 2 = 2x2 average)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-write existing EXR files")
    args = parser.parse_args()
    convert_all(
        rawhdr_root=args.rawhdr_root,
        splits=tuple(args.splits),
        data_range=args.data_range,
        downsample=args.downsample,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
