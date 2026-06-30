"""
Run all 5 HDR methods on the LTM split of a dataset and write EXR output.

Usage:
    python run_ltm_eval.py --dataset stuttgart [--gpu 0] [--methods hdrcnn lediff x2hdr lumivid ours]

Output layout:
    evaluations/{method}ltm_{dataset}/ltm/{video}/frame_XXXX.exr
"""
import argparse
import gc
import json
import os
import subprocess
import sys

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # hdrvideo/
DIFF_DIR   = os.path.dirname(os.path.abspath(__file__))                   # hdrvideo/diff/
EVAL_BASE  = os.path.join(DIFF_DIR, 'evaluations')
HDRCNN_DIR = os.path.join(BASE_DIR, 'hdrcnn')
LEDIFF_ROOT = os.path.join(BASE_DIR, 'lediff')
LEDIFF_HL  = os.path.join(BASE_DIR, 'lediff', 'model_highlight')
LEDIFF_SH  = os.path.join(BASE_DIR, 'lediff', 'model_shadow')
X2HDR_DIR  = os.path.join(BASE_DIR, 'X2HDRMay16')
LTX2_DIR   = os.path.join(BASE_DIR, 'LTX-2')
LTX2_DIST  = os.path.join(LTX2_DIR, 'models', 'ltx-2.3-22b-distilled-1.1.safetensors')
LTX2_UPS   = os.path.join(LTX2_DIR, 'models', 'ltx-2.3-spatial-upscaler-x2-1.1.safetensors')
LTX2_LORA  = os.path.join(LTX2_DIR, 'models', 'LTX-2.3-22b-IC-LoRA-HDR', 'ltx-2.3-22b-ic-lora-hdr-0.9.safetensors')
LTX2_EMB   = os.path.join(LTX2_DIR, 'models', 'LTX-2.3-22b-IC-LoRA-HDR', 'ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors')
INNER_SCRIPT = os.path.join(DIFF_DIR, 'run_ours_ltm_inner.py')
FPS = 24


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def in_dir(dataset, video):
    return os.path.join(EVAL_BASE, dataset, 'ltm', video)


def out_dir(method_tag, dataset, video):
    return os.path.join(EVAL_BASE, f'{method_tag}_{dataset}', 'ltm', video)


def ltm_videos(dataset):
    d = os.path.join(EVAL_BASE, dataset, 'ltm')
    if not os.path.isdir(d):
        raise FileNotFoundError(f'LTM dir not found: {d}')
    return sorted(v for v in os.listdir(d) if os.path.isdir(os.path.join(d, v)))


def is_done(o_dir, i_dir):
    if not os.path.isdir(o_dir):
        return False
    n_exr = sum(1 for f in os.listdir(o_dir) if f.endswith('.exr'))
    n_png = sum(1 for f in os.listdir(i_dir) if f.endswith('.png'))
    return n_png > 0 and n_exr >= n_png


def sorted_pngs(folder):
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith('.png') and not f.startswith('.'))


# ---------------------------------------------------------------------------
# HDRCNN
# ---------------------------------------------------------------------------

def run_hdrcnn(dataset):
    import torch

    _orig = os.getcwd()
    os.chdir(HDRCNN_DIR)
    sys.path.insert(0, HDRCNN_DIR)

    from model import HDRCNN
    from dataset_sdr import DatasetSDR
    from torch.utils.data import DataLoader
    from encode import decode_p
    from util import fromTorchToNP

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = HDRCNN().to(device)
    net.load_state_dict(torch.load('model.pth', map_location=device))
    net.eval()

    videos = ltm_videos(dataset)
    pending = [v for v in videos if not is_done(out_dir('hdrcnnltm', dataset, v), in_dir(dataset, v))]
    print(f'[hdrcnn] model loaded — {len(pending)}/{len(videos)} videos to process')

    for vid in pending:
        i_d = in_dir(dataset, vid)
        o_d = out_dir('hdrcnnltm', dataset, vid)
        os.makedirs(o_d, exist_ok=True)
        ds = DatasetSDR(i_d, device, True)
        dl = DataLoader(ds, batch_size=1)
        with torch.no_grad():
            for sdr, mask, sz_ori, sz_ab, fname in dl:
                fname = fname[0]
                sdr = sdr.to(device)
                mask = mask.to(device)
                out_lin = decode_p(net(sdr))
                out_np = torch.pow(sdr, 2.0) * (1.0 - mask) + mask * out_lin
                out_np = fromTorchToNP(out_np.detach().cpu().numpy().squeeze(0))
                out_np = out_np[0:sz_ori[0], 0:sz_ori[1], :]
                cv2.imwrite(os.path.join(o_d, fname + '.exr'),
                            cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR))
        print(f'[hdrcnn] {vid}: done')

    os.chdir(_orig)


# ---------------------------------------------------------------------------
# LEDiff
# ---------------------------------------------------------------------------

def run_lediff(dataset):
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from diffusers import StableDiffusionPipeline

    sys.path.insert(0, LEDIFF_ROOT)
    from src.diffusers import StableDiffusionHDRPipeline

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32

    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch_dtype
    ).to(device)
    vae = sd_pipe.vae.eval()
    hl_pipe = StableDiffusionHDRPipeline.from_pretrained(
        LEDIFF_HL, torch_dtype=torch.float32, model_path=LEDIFF_HL
    ).to(device)
    sh_pipe = StableDiffusionHDRPipeline.from_pretrained(
        LEDIFF_SH, torch_dtype=torch.float32, model_path=LEDIFF_SH
    ).to(device)
    tfm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    videos = ltm_videos(dataset)
    print(f'[lediff] models loaded — {len(videos)} videos')

    for vid in videos:
        i_d = in_dir(dataset, vid)
        o_d = out_dir('lediffltm', dataset, vid)
        if is_done(o_d, i_d):
            print(f'[lediff] {vid}: skipping')
            continue
        os.makedirs(o_d, exist_ok=True)
        for png in sorted_pngs(i_d):
            stem = os.path.splitext(png)[0]
            exr_path = os.path.join(o_d, stem + '.exr')
            if os.path.isfile(exr_path):
                continue
            img = Image.open(os.path.join(i_d, png))
            x = tfm(img.convert('RGB')).unsqueeze(0).to(device=device, dtype=torch_dtype)
            with torch.no_grad():
                z = vae.encode(x).latent_dist.mean.float().cpu().numpy()
                hl1, hl2, il = hl_pipe(output_type='latent', prompt='', latents_npy=z,
                                       height=img.height, width=img.width)
                sl1, sl2, il = sh_pipe(output_type='latent', prompt='', latents_npy=z,
                                       height=img.height, width=img.width)
                merged = sh_pipe.merge(hl1, hl2, il, None, device, sl1, sl2)
                result = sh_pipe.image_processor.postprocess(
                    merged, output_type='np', do_denormalize=[True])[0]
            hdr_bgr = cv2.cvtColor(
                np.exp(np.asarray(result)).astype(np.float32), cv2.COLOR_RGB2BGR)
            cv2.imwrite(exr_path, hdr_bgr)
        print(f'[lediff] {vid}: done')


# ---------------------------------------------------------------------------
# X2HDR (FLUX LDR→HDR LoRA)
# ---------------------------------------------------------------------------

def run_x2hdr(dataset):
    import torch
    from PIL import Image

    _orig = os.getcwd()
    os.chdir(X2HDR_DIR)
    sys.path.insert(0, X2HDR_DIR)
    # Clear any prior 'src' module (e.g. lediff's src.diffusers) from the cache
    for _k in list(sys.modules):
        if _k == 'src' or _k.startswith('src.'):
            sys.modules.pop(_k)

    from src.pipeline import FluxPipeline
    from src.transformer_flux import FluxTransformer2DModel
    from src.lora_helper import set_single_lora

    model_id  = os.path.join(X2HDR_DIR, 'models', 'Flux')
    lora_path = os.path.join(X2HDR_DIR, 'models', 'ldr2hdr_lora.safetensors')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, device=device)
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder='transformer', torch_dtype=torch.bfloat16, device=device)
    pipe.transformer = transformer
    pipe.to(device)

    # Determine frame resolution from first available frame
    videos = ltm_videos(dataset)
    first_png = sorted_pngs(in_dir(dataset, videos[0]))[0]
    sample = Image.open(os.path.join(in_dir(dataset, videos[0]), first_png))
    W, H = sample.width, sample.height
    set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_width=W, cond_height=H)
    print(f'[x2hdr] model loaded ({W}×{H}) — {len(videos)} videos')

    def _clear_cache():
        for _, ap in pipe.transformer.attn_processors.items():
            ap.bank_kv.clear()

    for vid in videos:
        i_d = in_dir(dataset, vid)
        o_d = out_dir('x2hdrltm', dataset, vid)
        if is_done(o_d, i_d):
            print(f'[x2hdr] {vid}: skipping')
            continue
        os.makedirs(o_d, exist_ok=True)
        for png in sorted_pngs(i_d):
            stem = os.path.splitext(png)[0]
            exr_path = os.path.join(o_d, stem + '.exr')
            if os.path.isfile(exr_path):
                continue
            img = Image.open(os.path.join(i_d, png)).convert('RGB').resize((W, H))
            with torch.inference_mode():
                _, hdr_image = pipe(
                    ' ', width=W, height=H,
                    guidance_scale=3.5, num_inference_steps=30,
                    max_sequence_length=512,
                    generator=torch.Generator('cpu').manual_seed(42),
                    spatial_images=[img], subject_images=[],
                    cond_width=W, cond_height=H,
                    hdr_mode=True, input_is_raw=False,
                )
            cv2.imwrite(exr_path, cv2.cvtColor(hdr_image.astype(np.float32), cv2.COLOR_RGB2BGR))
            _clear_cache()
        print(f'[x2hdr] {vid}: done')

    os.chdir(_orig)


# ---------------------------------------------------------------------------
# LumiVid (LTX-2 IC-LoRA)
# ---------------------------------------------------------------------------

def run_lumivid(dataset):
    import torch

    _orig = os.getcwd()
    os.chdir(LTX2_DIR)
    sys.path.insert(0, os.path.join(LTX2_DIR, 'packages', 'ltx-pipelines', 'src'))
    sys.path.insert(0, os.path.join(LTX2_DIR, 'packages', 'ltx-core', 'src'))

    from ltx_pipelines.hdr_ic_lora_png import HDRICLoraPipeline, _make_tiling_config
    from ltx_pipelines.utils.media_io import get_png_dir_metadata

    tiling_config = _make_tiling_config()
    pipeline = HDRICLoraPipeline(
        distilled_checkpoint_path=LTX2_DIST,
        spatial_upsampler_path=LTX2_UPS,
        hdr_lora=LTX2_LORA,
        text_embeddings_path=LTX2_EMB,
    )

    videos = ltm_videos(dataset)
    print(f'[lumivid] model loaded — {len(videos)} videos')

    for vid in videos:
        i_d = in_dir(dataset, vid)
        o_d = out_dir('lumividltm', dataset, vid)
        if is_done(o_d, i_d):
            print(f'[lumivid] {vid}: skipping')
            continue
        os.makedirs(o_d, exist_ok=True)
        pngs = sorted_pngs(i_d)
        meta = get_png_dir_metadata(i_d, fps=float(FPS))
        with torch.inference_mode():
            hdr_video = pipeline(
                seed=10,
                height=meta.height, width=meta.width,
                num_frames=len(pngs), frame_rate=float(FPS),
                video_conditioning=[(i_d, 1.0)],
                tiling_config=tiling_config,
            )
        for j, png in enumerate(pngs):
            stem = os.path.splitext(png)[0]
            frame_np = hdr_video[j].cpu().numpy().astype(np.float32)
            cv2.imwrite(os.path.join(o_d, stem + '.exr'),
                        cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
        del hdr_video
        gc.collect()
        torch.cuda.empty_cache()
        print(f'[lumivid] {vid}: done')

    os.chdir(_orig)


# ---------------------------------------------------------------------------
# Ours (WanVideo) — subprocess to isolate sys.path
# ---------------------------------------------------------------------------

def run_ours(dataset, gpu_id):
    videos = ltm_videos(dataset)
    jobs = [
        [in_dir(dataset, v), out_dir('oursltm', dataset, v)]
        for v in videos
        if not is_done(out_dir('oursltm', dataset, v), in_dir(dataset, v))
    ]
    if not jobs:
        print('[ours] all done, skipping')
        return

    for _, o_d in jobs:
        os.makedirs(o_d, exist_ok=True)

    env = os.environ.copy()  # CUDA_VISIBLE_DEVICES already set in os.environ
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')

    cmd = [sys.executable, INNER_SCRIPT, '--jobs', json.dumps(jobs)]
    print(f'[ours] launching subprocess for {len(jobs)} videos...', flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f'run_ours_ltm_inner.py exited with code {proc.returncode}')
    print('[ours] done')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['stuttgart', 'ubc'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--methods', nargs='+',
                        default=['hdrcnn', 'lediff', 'x2hdr', 'lumivid', 'ours'])
    args = parser.parse_args()

    # Honour externally-set CUDA_VISIBLE_DEVICES (e.g. from launcher script)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    method_fns = {
        'hdrcnn':  lambda: run_hdrcnn(args.dataset),
        'lediff':  lambda: run_lediff(args.dataset),
        'x2hdr':   lambda: run_x2hdr(args.dataset),
        'lumivid': lambda: run_lumivid(args.dataset),
        'ours':    lambda: run_ours(args.dataset, args.gpu),
    }

    for key in args.methods:
        if key not in method_fns:
            print(f'Unknown method: {key}')
            continue
        print(f'\n{"="*60}\nRunning {key} on {args.dataset}\n{"="*60}', flush=True)
        try:
            method_fns[key]()
        except Exception as exc:
            import traceback
            print(f'[ERROR] {key} failed: {exc}', flush=True)
            traceback.print_exc()


if __name__ == '__main__':
    main()
