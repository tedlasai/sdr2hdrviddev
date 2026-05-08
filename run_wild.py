import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HDRCNN_DIR = os.path.join(BASE_DIR, 'hdrcnn')
LEDIFF_DIR = os.path.join(BASE_DIR, 'lediff', 'examples', 'text_to_image')
LEDIFF_ROOT = os.path.join(BASE_DIR, 'lediff')
LEDIFF_HIGHLIGHT_MODEL = os.path.join(BASE_DIR, 'lediff', 'model_highlight')
LEDIFF_SHADOW_MODEL = os.path.join(BASE_DIR, 'lediff', 'model_shadow')
DIFF_DIR = os.path.join(BASE_DIR, 'diff')
OURS_CONFIG = os.path.join(BASE_DIR, 'diff', 'diffsynth', 'configs', 'threeexposures_crffixed_test_val.yaml')
LTX2_DIR       = os.path.join(BASE_DIR, 'LTX-2')
LTX2_DISTILLED = os.path.join(LTX2_DIR, 'models', 'ltx-2.3-22b-distilled-1.1.safetensors')
LTX2_UPSAMPLER = os.path.join(LTX2_DIR, 'models', 'ltx-2.3-spatial-upscaler-x2-1.1.safetensors')
LTX2_HDR_LORA  = os.path.join(LTX2_DIR, 'models', 'LTX-2.3-22b-IC-LoRA-HDR', 'ltx-2.3-22b-ic-lora-hdr-0.9.safetensors')
LTX2_TEXT_EMB  = os.path.join(LTX2_DIR, 'models', 'LTX-2.3-22b-IC-LoRA-HDR', 'ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors')
INPUT_DIR = os.path.join(BASE_DIR, 'wildvideos')
OUTPUT_DIR = os.path.join(BASE_DIR, 'wild_videos_out')

TARGET_W = 1280
TARGET_H = 704
N_FRAMES = 17
FPS = 24


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_frames(source, n, w, h):
    """Load n evenly-spaced frames from an mp4 or folder of images, resized to (w, h)."""
    if os.path.isfile(source):
        cap = cv2.VideoCapture(source)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total - 1, 0), n, dtype=int)
        frames = []
        last_good = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                last_good = cv2.resize(frame, (w, h))
            if last_good is not None:
                frames.append(last_good)
        cap.release()
    else:
        exts = ('.png', '.jpg', '.jpeg')
        files = sorted(f for f in os.listdir(source) if f.lower().endswith(exts))
        indices = np.linspace(0, max(len(files) - 1, 0), n, dtype=int)
        frames = []
        for idx in indices:
            img = cv2.imread(os.path.join(source, files[idx]))
            if img is not None:
                frames.append(cv2.resize(img, (w, h)))
    return frames


def save_frames(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(out_dir, f'{i:04d}.png'), frame)


def _write_mp4_via_ffmpeg(frames_iter, h, w, mp4_path):
    """Pipe BGR uint8 frames to ffmpeg for H.264 encoding."""
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(FPS),
        '-i', 'pipe:0',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', mp4_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frame in frames_iter:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def frames_to_mp4(frames_dir, mp4_path):
    """Build an H.264 mp4 from EXR frames in frames_dir, tonemapping each one."""
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.exr'))
    if not files:
        return
    first = tonemap_exr(os.path.join(frames_dir, files[0]))
    h, w = first.shape[:2]
    def _gen():
        yield first
        for f in files[1:]:
            yield tonemap_exr(os.path.join(frames_dir, f))
    _write_mp4_via_ffmpeg(_gen(), h, w, mp4_path)


def tonemap_exr(exr_path):
    img = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f'Cannot read EXR: {exr_path}')
    img = img.astype(np.float32)
    p99 = np.percentile(img, 99)
    if p99 > 0:
        img = img / p99
    img = img / (1.0 + img)  # Reinhard global
    img = np.power(np.clip(img, 0.0, 1.0), 1.0 / 2.2)
    return (img * 255).astype(np.uint8)


def exr_dir_to_png_dir(exr_dir, png_dir):
    for exr_file in sorted(f for f in os.listdir(exr_dir) if f.endswith('.exr')):
        stem = os.path.splitext(exr_file)[0]
        cv2.imwrite(os.path.join(png_dir, stem + '.png'),
                    tonemap_exr(os.path.join(exr_dir, exr_file)))


def is_done(frames_dir):
    if not os.path.isdir(frames_dir):
        return False
    return sum(1 for f in os.listdir(frames_dir) if f.endswith('.exr')) >= N_FRAMES


# ---------------------------------------------------------------------------
# Per-GPU workers — each loads its model(s) once, then iterates its video list
# ---------------------------------------------------------------------------

def hdrcnn_worker(jobs, gpu_id):
    """
    jobs: list of (name, input_frames_dir, output_frames_dir)
    Loads HDRCNN once, processes all jobs sequentially.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.chdir(HDRCNN_DIR)
    sys.path.insert(0, HDRCNN_DIR)

    import torch
    from model import HDRCNN
    from dataset_sdr import DatasetSDR
    from torch.utils.data import DataLoader
    from encode import decode_p
    from util import fromTorchToNP

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = HDRCNN().to(device)
    net.load_state_dict(torch.load('model.pth', map_location=device))
    net.eval()
    print(f'[GPU {gpu_id}] hdrcnn model loaded')

    for name, input_dir, output_dir in jobs:
        if is_done(output_dir):
            print(f'[GPU {gpu_id}] hdrcnn {name}: skipping')
            continue
        os.makedirs(output_dir, exist_ok=True)
        n_input = sum(1 for f in os.listdir(input_dir) if f.endswith('.png'))
        print(f'[GPU {gpu_id}] hdrcnn {name}: {n_input} input frames')

        ds = DatasetSDR(input_dir, device, True)
        dl = DataLoader(ds, batch_size=1)
        with torch.no_grad():
            for sdr, mask, sz_ori, sz_ab, fname in dl:
                fname = fname[0]
                sdr = sdr.to(device)
                mask = mask.to(device)
                sdr_hdr = net(sdr)
                out_lin = decode_p(sdr_hdr)
                out = torch.pow(sdr, 2.0) * (1.0 - mask) + mask * out_lin
                out = out.detach().cpu().numpy().squeeze(0)
                out_np = fromTorchToNP(out)
                out_np = out_np[0:sz_ori[0], 0:sz_ori[1], :]
                out_np = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, fname + '.exr'), out_np)

        frames_to_mp4(output_dir, os.path.join(OUTPUT_DIR, f'hdrcnn_{name}.mp4'))
        print(f'[GPU {gpu_id}] hdrcnn {name}: done')


def lediff_frames_worker(frame_jobs, gpu_id):
    """
    frame_jobs: list of (input_png_path, output_exr_path)
    Loads LEDiff models once, processes all assigned frames sequentially.
    Frame-level splitting lets all GPUs work even with few videos.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    sys.path.insert(0, LEDIFF_ROOT)

    import torch
    from PIL import Image
    from torchvision import transforms as T
    from diffusers import StableDiffusionPipeline
    from src.diffusers import StableDiffusionHDRPipeline

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32

    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch_dtype
    ).to(device)
    vae = sd_pipe.vae.eval()

    highlight_pipe = StableDiffusionHDRPipeline.from_pretrained(
        LEDIFF_HIGHLIGHT_MODEL, torch_dtype=torch.float32, model_path=LEDIFF_HIGHLIGHT_MODEL
    ).to(device)
    shadow_pipe = StableDiffusionHDRPipeline.from_pretrained(
        LEDIFF_SHADOW_MODEL, torch_dtype=torch.float32, model_path=LEDIFF_SHADOW_MODEL
    ).to(device)

    tfm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    print(f'[GPU {gpu_id}] lediff models loaded ({len(frame_jobs)} frames)')

    for input_path, output_path in frame_jobs:
        if os.path.isfile(output_path):
            continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.open(input_path)
        x = tfm(img.convert('RGB')).unsqueeze(0).to(device=device, dtype=torch_dtype)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mean.float().cpu().numpy()
        with torch.no_grad():
            hl1, hl2, il = highlight_pipe(
                output_type='latent', prompt='', latents_npy=z,
                height=img.height, width=img.width)
            sl1, sl2, il = shadow_pipe(
                output_type='latent', prompt='', latents_npy=z,
                height=img.height, width=img.width)
            merged = shadow_pipe.merge(hl1, hl2, il, None, device, sl1, sl2)
            result = shadow_pipe.image_processor.postprocess(
                merged, output_type='np', do_denormalize=[True])[0]
        hdr_bgr = cv2.cvtColor(np.exp(np.asarray(result)).astype(np.float32),
                               cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, hdr_bgr)
        print(f'[GPU {gpu_id}] lediff {os.path.basename(input_path)}: done')


def ours_worker(jobs, gpu_id):
    """
    jobs: list of (name, input_frames_dir, output_frames_dir)
    Loads our WanVideo model once, processes all jobs sequentially.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.chdir(DIFF_DIR)  # model paths resolve to ./models/ relative to diff/
    sys.path.insert(0, DIFF_DIR)
    sys.path.insert(0, os.path.join(DIFF_DIR, 'examples', 'wanvideo', 'model_training'))

    from accelerate import Accelerator
    from examples.wanvideo.model_training.train import (
        WanTrainingModule, load_yaml_config, set_load_paths,
    )
    from diffsynth.trainers.video_dataset import LoadPNGVideo
    from diffsynth.trainers.stuttgart_dataset import ImageCropAndResize
    from utils import output_frames
    from einops import rearrange
    import torch

    # Load config and model once
    import argparse
    args = argparse.Namespace(config=OURS_CONFIG)
    args = load_yaml_config(args, OURS_CONFIG)
    args = set_load_paths(args)

    accelerator = Accelerator()
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
        use_vae_ea=getattr(args, 'use_vae_ea', False),
    )
    model = accelerator.prepare(model)
    model.eval()
    model = accelerator.unwrap_model(model)
    print(f'[GPU {gpu_id}] ours model loaded')

    frame_processor = ImageCropAndResize(args.height, args.width, args.height * args.width, 16, 16)
    loader = LoadPNGVideo(num_frames=17, frame_processor=frame_processor)

    for name, input_dir, output_dir in jobs:
        if is_done(output_dir):
            print(f'[GPU {gpu_id}] ours {name}: skipping')
            continue
        os.makedirs(output_dir, exist_ok=True)
        print(f'[GPU {gpu_id}] ours {name}...')

        data = loader(input_dir)
        with torch.no_grad():
            outputs = model.pipe(
                prompt='',
                condition_video=data['input_video'],
                height=args.height,
                width=args.width,
                num_inference_steps=50,
                seed=1, tiled=False, cfg_scale=1.0,
                encoder_decoder_mode=model.encoder_decoder_mode,
                exposures=data['exposures'],
                generate_exposures=(-4, 0, 4),
                use_vae_ea=model.use_vae_ea,
            )

        out = rearrange(outputs['hdr_video'], 'b c t h w -> b t c h w')

        output_frames(out[0], output_dir, mode='hdr', channel_order='NCHW')

        frames_to_mp4(output_dir, os.path.join(OUTPUT_DIR, f'ours_{name}.mp4'))
        print(f'[GPU {gpu_id}] ours {name}: done')


def lumivid_worker(jobs, gpu_id):
    """
    jobs: list of (name, input_frames_dir, output_frames_dir)
    Loads LTX-2 HDR IC-LoRA pipeline once, processes all jobs sequentially.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.chdir(LTX2_DIR)  # model paths resolve to ./models/ relative to LTX-2/
    sys.path.insert(0, os.path.join(LTX2_DIR, 'packages', 'ltx-pipelines', 'src'))
    sys.path.insert(0, os.path.join(LTX2_DIR, 'packages', 'ltx-core', 'src'))

    import gc
    import torch
    from ltx_pipelines.hdr_ic_lora_png import HDRICLoraPipeline, _make_tiling_config
    from ltx_pipelines.utils.media_io import get_png_dir_metadata

    tiling_config = _make_tiling_config()
    pipeline = HDRICLoraPipeline(
        distilled_checkpoint_path=LTX2_DISTILLED,
        spatial_upsampler_path=LTX2_UPSAMPLER,
        hdr_lora=LTX2_HDR_LORA,
        text_embeddings_path=LTX2_TEXT_EMB,
    )
    print(f'[GPU {gpu_id}] lumivid model loaded')

    for name, input_dir, output_dir in jobs:
        if is_done(output_dir):
            print(f'[GPU {gpu_id}] lumivid {name}: skipping')
            continue
        os.makedirs(output_dir, exist_ok=True)
        print(f'[GPU {gpu_id}] lumivid {name}...')

        meta = get_png_dir_metadata(input_dir, fps=float(FPS))
        with torch.inference_mode():
            hdr_video = pipeline(
                seed=10,
                height=meta.height,
                width=meta.width,
                num_frames=N_FRAMES,
                frame_rate=float(FPS),
                video_conditioning=[(input_dir, 1.0)],
                tiling_config=tiling_config,
            )

        # hdr_video: [f, h, w, c] float32 RGB linear HDR → save as BGR EXR
        for j in range(hdr_video.shape[0]):
            frame_np = hdr_video[j].cpu().numpy().astype(np.float32)
            frame_bgr = frame_np[:, :, ::-1].copy()
            cv2.imwrite(os.path.join(output_dir, f'frame_{j:05d}.exr'), frame_bgr)

        del hdr_video
        gc.collect()
        torch.cuda.empty_cache()

        frames_to_mp4(output_dir, os.path.join(OUTPUT_DIR, f'lumivid_{name}.mp4'))
        print(f'[GPU {gpu_id}] lumivid {name}: done')


def gpu_worker(gpu_id, video_chunk):
    """Entry point for each GPU process. Handles input extraction, hdrcnn, then lediff."""
    # Input extraction (CPU only)
    for name, source in video_chunk:
        input_frames_dir = os.path.join(OUTPUT_DIR, f'input_{name}')
        input_mp4 = os.path.join(OUTPUT_DIR, f'input_{name}.mp4')
        png_done = (os.path.isdir(input_frames_dir) and
                    sum(1 for f in os.listdir(input_frames_dir) if f.endswith('.png')) >= N_FRAMES)
        if not png_done:
            frames = load_frames(source, N_FRAMES, TARGET_W, TARGET_H)
            print(f'[GPU {gpu_id}] input {name}: {len(frames)} frames')
            save_frames(frames, input_frames_dir)
        else:
            print(f'[GPU {gpu_id}] input {name}: skipping frame extraction')
        if not os.path.isfile(input_mp4):
            png_files = sorted(f for f in os.listdir(input_frames_dir) if f.endswith('.png'))
            if png_files:
                first = cv2.imread(os.path.join(input_frames_dir, png_files[0]))
                h, w = first.shape[:2]
                def _png_gen():
                    yield first
                    for f in png_files[1:]:
                        yield cv2.imread(os.path.join(input_frames_dir, f))
                _write_mp4_via_ffmpeg(_png_gen(), h, w, input_mp4)

    def jobs_for(method):
        return [
            (name, os.path.join(OUTPUT_DIR, f'input_{name}'), os.path.join(OUTPUT_DIR, f'{method}_{name}'))
            for name, _ in video_chunk
        ]

    def run_if_needed(worker_fn, method):
        jobs = jobs_for(method)
        pending = [j for j in jobs if not is_done(j[2])]
        if not pending:
            print(f'[GPU {gpu_id}] {method}: all done, skipping')
            return
        worker_fn(pending, gpu_id)

    run_if_needed(ours_worker, 'ours')
    run_if_needed(lumivid_worker, 'lumivid')
    run_if_needed(hdrcnn_worker, 'hdrcnn')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', default=list(range(8)),
                        help='GPU IDs to use (default: 0-7)')
    args = parser.parse_args()

    tmp_dir = os.path.join(BASE_DIR, '.tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ['TMPDIR'] = tmp_dir
    triton_dir = os.path.join(BASE_DIR, '.triton')
    os.makedirs(triton_dir, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = triton_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    videos = []
    for entry in sorted(os.listdir(INPUT_DIR)):
        if entry.startswith('.'):
            continue
        path = os.path.join(INPUT_DIR, entry)
        name = os.path.splitext(entry)[0] if os.path.isfile(path) else entry
        videos.append((name, path))

    # Slice videos into contiguous chunks, one per GPU
    n_gpus = len(args.gpus)
    chunks = [videos[i::n_gpus] for i in range(n_gpus)]

    # Only spawn processes for GPUs that actually have work
    active = [(gpu, chunk) for gpu, chunk in zip(args.gpus, chunks) if chunk]

    # Phase 1: input extraction + ours + lumivid + hdrcnn (split by video)
    with ProcessPoolExecutor(max_workers=len(active)) as ex:
        futures = {ex.submit(gpu_worker, gpu, chunk): gpu for gpu, chunk in active}
        for fut in as_completed(futures):
            gpu = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f'[FAIL GPU {gpu}]: {e}')

    # Phase 2: LEDiff — split all frames across all GPUs
    all_frame_jobs = []
    for name, _ in videos:
        input_dir = os.path.join(OUTPUT_DIR, f'input_{name}')
        output_dir = os.path.join(OUTPUT_DIR, f'lediff_{name}')
        if not os.path.isdir(input_dir):
            continue
        for f in sorted(f for f in os.listdir(input_dir) if f.endswith('.png')):
            out = os.path.join(output_dir, os.path.splitext(f)[0] + '.exr')
            if not os.path.isfile(out):
                all_frame_jobs.append((os.path.join(input_dir, f), out))

    if all_frame_jobs:
        print(f'LEDiff: {len(all_frame_jobs)} frames across {n_gpus} GPUs')
        frame_chunks = [all_frame_jobs[i::n_gpus] for i in range(n_gpus)]
        active_lediff = [(gpu, chunk) for gpu, chunk in zip(args.gpus, frame_chunks) if chunk]
        with ProcessPoolExecutor(max_workers=len(active_lediff)) as ex:
            futures = {ex.submit(lediff_frames_worker, chunk, gpu): gpu
                       for gpu, chunk in active_lediff}
            for fut in as_completed(futures):
                gpu = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f'[FAIL GPU {gpu}] lediff: {e}')

        # Create LEDiff MP4s now that all frames are done
        for name, _ in videos:
            mp4_path = os.path.join(OUTPUT_DIR, f'lediff_{name}.mp4')
            frames_dir = os.path.join(OUTPUT_DIR, f'lediff_{name}')
            if not os.path.isfile(mp4_path) and is_done(frames_dir):
                frames_to_mp4(frames_dir, mp4_path)
                print(f'lediff {name}: mp4 done')
    else:
        print('LEDiff: all frames already done')


if __name__ == '__main__':
    main()
