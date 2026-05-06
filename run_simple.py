import torch
import sys
sys.path.append("/data2/saikiran.tedla/hdrvideo/DiffSynth-Studio-main/diffsynth")

from dotenv import load_dotenv
load_dotenv()

from diffsynth_og import save_video
from diffsynth_og.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

# pipe = WanVideoPipeline.from_pretrained(
#     torch_dtype=torch.bfloat16,
#     device="cuda",
#     model_configs=[
#         # ModelConfig(path="/data2/saikiran.tedla/hdrvideo/diff/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", skip_download=True),
#         # ModelConfig(path="/data2/saikiran.tedla/hdrvideo/diff/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
#         # ModelConfig(path="/data2/saikiran.tedla/hdrvideo/diff/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", skip_download=True),
       
#         ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"), #skip_download=True),
#         ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"), #skip_download=True),
#         ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"), #skip_download=True),
  
#     ],
#     use_usp=False
# )

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    redirect_common_files=False,
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", skip_download=True),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu", skip_download=True),
    ],
)

pipe.enable_vram_management()

#prompts = [{"name": "brighter.mp4", "prompt": "A puppy running over the grass in slow motion. The puppy has brown-yellow fur, upright ears, and looks focused and joyful. The scene gradually becomes brighter over time, as if the exposure is slowly increasing frame by frame."},
#           {"name": "darker.mp4", "prompt": "A puppy running over the grass in slow motion. The puppy has brown-yellow fur, upright ears, and looks focused and joyful. The scene gradually becomes darker over time, as if the exposure is slowly decreasing frame by frame."}]
# prompts = [{
#     "name": "brighter.mp4",
#         "prompt": (
#         "A puppy running over the grass in slow motion. "
#         "The first frames appear dim and underexposed, with muted colors and low contrast. "
#         "As the video progresses, the scene becomes dramatically brighter, ending in intense, high-exposure lighting "
#         "with blown-out highlights, washed-out whites, and strong glare from the sun."
#     )
# },
# {
#     "name": "brighter_stil.mp4",
#         "prompt": (
#         "A puppy sitting still on the grass in slow motion. "
#         "The first frames appear dim and underexposed, with muted colors and low contrast. "
#         "As the video progresses, the scene becomes dramatically brighter, ending in intense, high-exposure lighting "
#         "with blown-out highlights, washed-out whites, and strong glare from the sun."
#     )
# },
#   {"name": "spotlight_brightening.mp4",
#   "prompt": (
#     "A single performer standing on a dark stage. "
#     "The scene begins almost completely dark, with only faint outlines visible. "
#     "Over time, powerful stage spotlights turn on and intensify, "
#     "flooding the stage with extremely bright light, causing strong glare, "
#     "lens flare, and blown-out highlights by the final frames."
#   ),
#   },{
#     "name": "tunnel_exit.mp4",
#   "prompt": (
#     "A camera moving through a dark tunnel toward an exit. "
#     "The interior appears dim and underexposed. "
#     "As the camera approaches the tunnel opening, "
#     "intense sunlight pours in, rapidly increasing brightness until "
#     "the exterior becomes extremely bright and partially overexposed."
#   ),
#     },{
#     "name": "sunrise_city.mp4",
#   "prompt": (
#     "A static shot of a city skyline before sunrise. "
#     "The scene starts very dark with silhouettes of buildings. "
#     "As the sun rises, the sky rapidly brightens, flooding the scene with "
#     "intense morning light, increasing exposure and washing out the sky "
#     "by the end of the sequence."
#   ),
#     },{
#     "name": "lights_on.mp4",
#   "prompt": (
#     "A dimly lit room with furniture barely visible. "
#     "Suddenly, the lights switch on and continue to brighten, "
#     "making the room extremely bright with flat lighting and clipped highlights."
#   )
# }   
# ]


prompts = [
  {
    "name": "puppy_fading_dark.mp4",
    "prompt": (
      "A puppy sitting still on the grass in slow motion. "
      "The first frames appear bright and well-exposed, with vibrant colors and clear detail. "
      "As the video progresses, the scene gradually becomes darker and more underexposed, "
      "ending in extremely dim lighting with crushed blacks, muted colors, and very low visibility."
    )
  },
  {
    "name": "stage_lights_fading_out.mp4",
    "prompt": (
      "A single performer standing on a stage under strong spotlights. "
      "The scene begins brightly lit, with clear visibility of the performer and stage. "
      "Over time, the stage lights slowly dim and fade out, "
      "plunging the stage into darkness until only faint outlines remain, "
      "ending in near-total blackness with heavy shadow and loss of detail."
    )
  },
  {
    "name": "sunset_city_darkening.mp4",
    "prompt": (
      "A static shot of a city skyline during sunset. "
      "The scene starts brightly lit with warm sunlight reflecting off buildings. "
      "As the sun sets, the sky darkens rapidly and the city becomes increasingly shadowed, "
      "ending in deep nighttime darkness with silhouettes of buildings and underexposed streets."
    )
  },
  {
    "name": "forest_light_fading.mp4",
    "prompt": (
      "A wide shot of a forest with sunlight shining through the trees. "
      "The first frames are bright and richly detailed, with strong highlights and vibrant greens. "
      "As the video progresses, the light slowly fades as if clouds cover the sun, "
      "ending in very dim lighting with low contrast, muted colors, and heavy underexposure."
    )
  },
  {
    "name": "room_lights_turning_off.mp4",
    "prompt": (
      "A well-lit room with furniture clearly visible. "
      "Suddenly, the lights begin to dim and continue darkening, "
      "making the room progressively harder to see, "
      "ending in extreme darkness with crushed shadows and barely visible objects."
    )
  },
  {
    "name": "street_day_to_night.mp4",
    "prompt": (
      "A static shot of a busy street during the daytime. "
      "The scene begins bright and well-exposed with clear detail in buildings and cars. "
      "As the video progresses, the lighting fades as if transitioning into night, "
      "ending in very dark exposure with only a few dim streetlights visible and most details lost in shadow."
    )
  },
  {
    "name": "beach_sunlight_fading.mp4",
    "prompt": (
      "A calm beach scene with waves rolling in under strong sunlight. "
      "The first frames appear bright and clear with strong reflections on the water. "
      "As the video progresses, the sunlight fades away and the scene becomes increasingly dark, "
      "ending in deep underexposure with muted colors and the ocean barely visible."
    )
  },
  {
    "name": "car_headlights_shutting_off.mp4",
    "prompt": (
      "A car parked on a road at night with bright headlights illuminating the scene. "
      "The first frames are clearly visible due to strong artificial lighting. "
      "As the video progresses, the headlights gradually dim and fade out, "
      "ending in near-total darkness with only faint outlines of the car visible."
    )
  },
  {
    "name": "tunnel_entering_darkness.mp4",
    "prompt": (
      "A camera moving from bright daylight into a tunnel entrance. "
      "The exterior begins well-exposed with strong sunlight and clear detail. "
      "As the camera enters the tunnel, the scene rapidly becomes darker and underexposed, "
      "ending deep inside the tunnel with extreme darkness, crushed blacks, and minimal visibility."
    )
  },
  {
    "name": "lamp_flicker_to_black.mp4",
    "prompt": (
      "A small desk lamp lighting a dark room. "
      "The first frames show the lamp shining brightly, clearly illuminating nearby objects. "
      "As the video progresses, the lamp begins flickering and slowly fading, "
      "ending in complete darkness with the scene nearly black and all details lost."
    )
  }
]


for item in prompts:
    video = pipe(
        prompt=item["prompt"],
        negative_prompt="static, blurry details, subtitles, style, artwork, image, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed limbs, fused fingers, still frame, messy background, three legs, crowded background people, walking backwards",
        seed=0,
        tiled = False,
        num_frames = 81,
        # num_inference_steps = 10,
        height = 704,
        width = 1280,
    )
    save_video(video, item["name"], fps=15, quality=5)
    print(f"Saved video: {item['name']}")

# Image-to-video
# from PIL import Image
# input_image = Image.open("/data2/saikiran.tedla/hdrvideo/diff/dog.jpg").resize((1248, 704))
# video = pipe(
#     prompt="make the exposure darker over time",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=False,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=17,
# )
# save_video(video, "darker.mp4", fps=15, quality=5)

# video = pipe(
#     prompt="make the lighting brighter over time",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=False,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=17,
# )
#save_video(video, "brighter.mp4", fps=15, quality=5)