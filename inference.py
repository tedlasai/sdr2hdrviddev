"""
Single-video HDR inference for our WanVideo-based method.

Usage:
    python diff/inference.py \
        --input_dir  wildvideos/input_x  \
        --output_dir wild_videos_out/ours_x_exr \
        [--config    diff/diffsynth/configs/threeexposures_crffixed_test_val.yaml]

Output: EXR files (frame_0000.exr, ...) written to --output_dir.
Tonemapping / mp4 creation is handled by the caller (run_wild.py).
"""

import argparse
import os
import sys

# Make diff/ and model_training/ importable regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'examples', 'wanvideo', 'model_training'))

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import torch
import numpy as np
from einops import rearrange

from examples.wanvideo.model_training.train import (
    WanTrainingModule, load_yaml_config, set_load_paths,
)
from diffsynth.trainers.video_dataset import LoadPNGVideo
from diffsynth.trainers.stuttgart_dataset import ImageCropAndResize
from utils import output_frames

DEFAULT_CONFIG = os.path.join(
    _HERE, 'diffsynth', 'configs', 'threeexposures_crffixed_test_val.yaml'
)


def build_model(config_path):
    """Load config and return an initialised WanTrainingModule (on CPU)."""
    args = argparse.Namespace(config=config_path)
    args = load_yaml_config(args, config_path)
    args = set_load_paths(args)

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
    return model, args


def run_inference(model, args, input_dir, output_dir):
    """Run the pipe on one video folder and write EXRs to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    frame_processor = ImageCropAndResize(
        args.height, args.width, args.height * args.width, 16, 16
    )
    loader = LoadPNGVideo(num_frames=17, frame_processor=frame_processor)
    data = loader(input_dir)

    condition_video = data['input_video']   # (T, H, W, C) uint8
    exposures = data['exposures']           # (4,) float

    with torch.no_grad():
        outputs = model.pipe(
            prompt='',
            condition_video=condition_video,
            height=args.height,
            width=args.width,
            num_inference_steps=50,
            seed=1,
            tiled=False,
            cfg_scale=1.0,
            encoder_decoder_mode=model.encoder_decoder_mode,
            exposures=exposures,
            generate_exposures=(-4, 0, 4),
            use_vae_ea=model.use_vae_ea,
        )

    # outputs["hdr_video"]: (b, c, t, h, w)
    out = rearrange(outputs['hdr_video'], 'b c t h w -> b t c h w')
    output_frames(out[0], output_dir, mode='hdr', channel_order='NCHW')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    cli = parser.parse_args()

    from accelerate import Accelerator
    accelerator = Accelerator()

    model, args = build_model(cli.config)
    model = accelerator.prepare(model)
    model.eval()
    model = accelerator.unwrap_model(model)

    run_inference(model, args, cli.input_dir, cli.output_dir)
    print(f'Done → {cli.output_dir}')


if __name__ == '__main__':
    main()
