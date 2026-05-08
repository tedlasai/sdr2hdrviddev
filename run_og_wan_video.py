import torch
import sys
sys.path.append("/data2/saikiran.tedla/hdrvideo/DiffSynth-Studio-main/diffsynth")

from dotenv import load_dotenv
load_dotenv()

from diffsynth_og import save_video
from diffsynth_og.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


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
