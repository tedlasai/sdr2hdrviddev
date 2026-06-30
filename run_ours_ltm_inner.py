"""
Subprocess worker: load the Ours WanVideo model once, then process a list of
(input_dir, output_dir) pairs, writing frame_XXXX.exr to each output_dir.

Called by run_ltm_eval.py.  Never imported directly.

Usage:
    python run_ours_ltm_inner.py --jobs '[["in1","out1"],["in2","out2"]]'
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'examples', 'wanvideo', 'model_training'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np
import torch
from accelerate import Accelerator
from einops import rearrange

from examples.wanvideo.model_training.train import WanTrainingModule, load_yaml_config, set_load_paths
from diffsynth.trainers.video_dataset import LoadPNGVideo
from diffsynth.trainers.stuttgart_dataset import ImageCropAndResize
from utils import output_frames

DEFAULT_CONFIG = os.path.join(_HERE, 'diffsynth', 'configs', 'threeexposures_crffixed_test_val.yaml')
DECODER_PATH   = os.path.join(_HERE, 'models', 'train',
                              'three_exposures_crfchanging_multimode_17_val',
                              'checkpoints_final', 'mergedeocderwieghts.safetensors')

parser = argparse.ArgumentParser()
parser.add_argument('--jobs', required=True, help='JSON list of [input_dir, output_dir] pairs')
parser.add_argument('--config', default=DEFAULT_CONFIG)
args = parser.parse_args()

jobs = json.loads(args.jobs)

# ---------------------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------------------
_cfg = argparse.Namespace(config=args.config)
_cfg = load_yaml_config(_cfg, args.config)
_cfg.decoder_path = DECODER_PATH
_cfg = set_load_paths(_cfg)

accelerator = Accelerator()
model = WanTrainingModule(
    model_paths=_cfg.model_paths,
    model_id_with_origin_paths=_cfg.model_id_with_origin_paths,
    trainable_models=_cfg.trainable_models,
    lora_base_model=_cfg.lora_base_model,
    lora_target_modules=_cfg.lora_target_modules,
    lora_rank=_cfg.lora_rank,
    lora_checkpoint=_cfg.lora_checkpoint,
    use_gradient_checkpointing_offload=_cfg.use_gradient_checkpointing_offload,
    extra_inputs=_cfg.extra_inputs,
    max_timestep_boundary=_cfg.max_timestep_boundary,
    min_timestep_boundary=_cfg.min_timestep_boundary,
    encoder_decoder_mode=_cfg.encode_decoder_mode,
    use_vae_ea=getattr(_cfg, 'use_vae_ea', False),
)
model = accelerator.prepare(model)
model.eval()
model = accelerator.unwrap_model(model)
print('[ours] model loaded', flush=True)

frame_processor = ImageCropAndResize(_cfg.height, _cfg.width, _cfg.height * _cfg.width, 16, 16)
loader = LoadPNGVideo(num_frames=_cfg.num_hdr_frames, frame_processor=frame_processor)

# ---------------------------------------------------------------------------
# Process each job
# ---------------------------------------------------------------------------
for input_dir, output_dir in jobs:
    n_expected = len([f for f in os.listdir(input_dir) if f.endswith('.png')])
    n_done = len([f for f in os.listdir(output_dir) if f.endswith('.exr')]) if os.path.isdir(output_dir) else 0
    if n_done >= n_expected:
        print(f'[ours] skip (done): {output_dir}', flush=True)
        continue

    print(f'[ours] {os.path.basename(input_dir)} → {output_dir}', flush=True)
    os.makedirs(output_dir, exist_ok=True)

    data = loader(input_dir)
    with torch.no_grad():
        outputs = model.pipe(
            prompt='',
            condition_video=data['input_video'],
            height=_cfg.height,
            width=_cfg.width,
            num_inference_steps=50,
            seed=1, tiled=False, cfg_scale=1.0,
            encoder_decoder_mode=model.encoder_decoder_mode,
            exposures=data['exposures'],
            generate_exposures=(-4, 0, 4),
            use_vae_ea=model.use_vae_ea,
        )

    hdr = rearrange(outputs['hdr_video'], 'b c t h w -> b t c h w')
    output_frames(hdr[0], output_dir, mode='hdr', channel_order='NCHW')
    del outputs, hdr
    torch.cuda.empty_cache()
    print(f'[ours] done: {output_dir}', flush=True)

print('OURS_INNER_DONE', flush=True)
