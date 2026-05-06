import torch
import sys
sys.path.append("/data2/saikiran.tedla/hdrvideo/diff")


from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

from diff.diffsynth.trainers.stuttgart_dataset import StuttgartDataset





dataset = StuttgartDataset(
    base_path="/data2/saikiran.tedla/hdrvideo/diff/data/stuttgart/carousel_fireworks_02",
    repeat=1,
    main_data_operator=StuttgartDataset.default_video_operator(
        base_path="/data2/saikiran.tedla/hdrvideo/diff/data/stuttgart/carousel_fireworks_02",
        max_pixels=1280*720,
        height=480,
        width=832,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=15,
        time_division_factor=4,
        time_division_remainder=1,
    ),
)

# Get a sample from the dataset
data = dataset[0]
condition_video = data["video"]
#import numpy as np
#save_video(condition_video, "stuttgart_input.mp4", fps=15)

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", skip_download=True),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu", skip_download=True),
    ],
)
state_dict = load_state_dict("/data2/saikiran.tedla/hdrvideo/diff/models/train/three_exposures/checkpoints/epoch-19.safetensors")
pipe.dit.load_state_dict(state_dict)
pipe.enable_vram_management()

condition_video = data["video"]

video = pipe(
    prompt="",
    #negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    condition_video=condition_video[0:5],  # Use the first 5 frames as condition
    num_frames=15,
    num_inference_steps=50,
    seed=1, tiled=False,
    cfg_scale=1.0
)
print("VAE OUTPUT SHAPE:", video.shape)
pil_video = pipe.vae_output_to_video(video)
print("num frames:", len(pil_video))
save_video(pil_video, "stuttgart_output.mp4", fps=15, quality=10)
# write out the frames individually to a folder