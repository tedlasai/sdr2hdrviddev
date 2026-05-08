import argparse
import hashlib
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch


def _repo_path(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, *parts))


def _sanitize_filename(s: str, max_len: int = 140) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(",", "_")
    s = re.sub(r"[^a-zA-Z0-9._ -]+", "_", s).strip(" ._-")
    s = s.replace(" ", "_")
    if not s:
        s = "prompt"
    if len(s) <= max_len:
        return s
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"{s[: max_len - 11].rstrip(' ._-')}_{h}"


def _first_sentence(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        return prompt
    m = re.search(r"[.!?]", prompt)
    if not m:
        return prompt
    return prompt[: m.start()]


def _read_prompts(path: str) -> list[str]:
    prompts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
    return prompts


def build_pipe(device: str, dtype: torch.dtype):
    # Match the import behavior of diff/run_og_wan_video.py
    diffsynth_dir = "/data2/saikiran.tedla/hdrvideo/DiffSynth-Studio-main/diffsynth"
    if diffsynth_dir not in sys.path:
        sys.path.append(diffsynth_dir)

    from diffsynth_og.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        redirect_common_files=False,
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
                skip_download=True,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
                skip_download=True,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="Wan2.2_VAE.pth",
                offload_device="cpu",
                skip_download=True,
            ),
        ],
    )
    pipe.enable_vram_management()
    return pipe


def _worker(gpu_id: int, prompts: list[str], out_dir: str, args_dict: dict):
    # Each worker is a separate process; bind it to one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Reconstruct args (can't pass argparse.Namespace reliably across mp start methods)
    num_frames = int(args_dict["num_frames"])
    fps = int(args_dict["fps"])
    quality = int(args_dict["quality"])
    height = int(args_dict["height"])
    width = int(args_dict["width"])
    negative_prompt = str(args_dict["negative_prompt"])
    seed = int(args_dict["seed"])
    save_frames = bool(args_dict.get("save_frames", False))
    frames_dir_name = str(args_dict.get("frames_dir_name", "frames"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = build_pipe(device=device, dtype=dtype)

    from diffsynth_og import save_video
    from diffsynth_og.data.video import save_frames

    results: list[tuple[str, str]] = []
    for prompt in prompts:
        stem = _sanitize_filename(_first_sentence(prompt))
        out_path = os.path.join(out_dir, f"{stem}.mp4")
        if os.path.isfile(out_path):
            results.append((out_path, "skip"))
            continue

        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            tiled=False,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        save_video(video, out_path, fps=fps, quality=quality)
        if save_frames:
            frames_dir = os.path.join(out_dir, frames_dir_name, stem)
            save_frames(video, frames_dir)
        results.append((out_path, "ok"))
    return gpu_id, results


def main():
    parser = argparse.ArgumentParser(description="Generate Wan videos from a prompts file (multi-GPU).")
    parser.add_argument(
        "--prompts_file",
        required=True,
        help="Text file with one prompt per line (# for comments).",
    )
    parser.add_argument(
        "--out_dir",
        default=_repo_path("genvideos"),
        help="Output directory for mp4s (default: diff/genvideos).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=list(range(8)),
        help="GPU IDs to use (default: 0-7).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fixed seed for determinism (not randomized).",
    )
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate.")
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS.")
    parser.add_argument("--quality", type=int, default=5, help="Save quality (diffsynth_og.save_video).")
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Also save per-frame PNGs under <out_dir>/<frames_dir_name>/<video_stem>/ (default: off).",
    )
    parser.add_argument(
        "--frames_dir_name",
        default="frames",
        help="Subdirectory name inside out_dir for dumped frames (default: frames).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Frame height. (Default is 704; this matches existing scripts.)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Frame width. (Default is 1280; this matches existing scripts.)",
    )
    parser.add_argument(
        "--negative_prompt",
        default=(
            "static, blurry details, subtitles, style, artwork, image, still, overall gray, "
            "worst quality, low quality, JPEG compression artifacts, ugly, deformed, extra fingers, "
            "poorly drawn hands, poorly drawn face, malformed limbs, fused fingers, still frame, "
            "messy background, three legs, crowded background people, walking backwards"
        ),
        help="Negative prompt.",
    )
    args = parser.parse_args()

    prompts = _read_prompts(args.prompts_file)
    if not prompts:
        raise SystemExit(f"No prompts found in: {args.prompts_file}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Split prompts across GPUs (one process per GPU), each process loads the model once.
    gpus = list(args.gpus)
    if not gpus:
        raise SystemExit("--gpus must have at least one GPU id")

    chunks = [prompts[i:: len(gpus)] for i in range(len(gpus))]
    active = [(gpu, chunk) for gpu, chunk in zip(gpus, chunks) if chunk]

    args_dict = {
        "num_frames": args.num_frames,
        "fps": args.fps,
        "quality": args.quality,
        "height": args.height,
        "width": args.width,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "save_frames": args.save_frames,
        "frames_dir_name": args.frames_dir_name,
    }

    with ProcessPoolExecutor(max_workers=len(active)) as ex:
        futs = {
            ex.submit(_worker, gpu, chunk, args.out_dir, args_dict): gpu
            for gpu, chunk in active
        }
        for fut in as_completed(futs):
            gpu = futs[fut]
            try:
                gpu_id, results = fut.result()
            except Exception as e:
                print(f"[FAIL GPU {gpu}]: {e}")
                continue
            n_ok = sum(1 for _, s in results if s == "ok")
            n_skip = sum(1 for _, s in results if s == "skip")
            print(f"[GPU {gpu_id}] done: ok={n_ok} skip={n_skip}")


if __name__ == "__main__":
    main()

