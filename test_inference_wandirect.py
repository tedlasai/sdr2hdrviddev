import torch
import sys
sys.path.append("/data2/saikiran.tedla/hdrvideo/DiffSynth-Studio-main/diffsynth")

from dotenv import load_dotenv
load_dotenv()
import torch
from PIL import Image
from diffsynth_og import save_video
from diffsynth_og.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

#model_id_with_origin_paths: "Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", skip_download=True),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu", skip_download=True)
    ],
)
pipe.enable_vram_management()

# Text-to-video
video = pipe(
    prompt="A documentary photography style scene: a lively puppy rapidly running on green grass. The puppy has brown-yellow fur, upright ears, and looks focused and joyful. Sunlight shines on its body, making the fur appear soft and shiny. The background is an open field with occasional wildflowers, and faint blue sky and clouds in the distance. Strong sense of perspective captures the motion of the puppy and the vitality of the surrounding grass. Mid-shot side-moving view.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=False,
    height=704, width=1248,
    num_frames=17,
)
save_video(video, "video1.mp4", fps=15, quality=5)

# # Image-to-video
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
# )
# input_image = Image.open("data/examples/wan/cat_fightning.jpg").resize((1248, 704))
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=False,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=17,
# )
# save_video(video, "video2.mp4", fps=15, quality=5)