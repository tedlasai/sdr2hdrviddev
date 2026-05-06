import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
import time

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models.model_manager import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_vae_merge_decoder import WanVideoVAEMergeDecoder
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader
from .wan_video_extra import TemporalTiler_BCTHW, model_fn_wan_video
from einops import reduce


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None, 
            vae = None, dit = None, scheduler = None, image_encoder = None, text_encoder = None,merge_decoder = None,
            prompter = None, input_type="normal"):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = scheduler if scheduler is not None else FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = prompter if prompter is not None else WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = text_encoder
        self.image_encoder: WanImageEncoder = image_encoder
        self.dit: WanModel = dit
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = vae
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.merge_decoder = merge_decoder
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_CustomVideoConditionEmbedderFused(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = [
        ]
        self.model_fn = model_fn_wan_video
        
    
    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

        
    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        # timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        # timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        #Diffusion forcing
        #forcing_options = ["none"]#, "split"]
        #choose 50% of the time to do none and 50% of the time to do split
        forcing_type = "none" #np.random.choice(forcing_options, p=[0.5, 0.5]) #(2nd stage)
        #forcing_type = "none" (1st stage)
        num_latents = inputs["input_latents"].shape[2]
        if forcing_type == "full":
            timestep_ids = torch.randint(min_timestep_boundary,max_timestep_boundary, (num_latents,))
        elif forcing_type == "split":
            history_timestep = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
            new_timestep = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))

            timestep_ids = torch.full((num_latents,), new_timestep.item())

            # create mask: True for positions 0,1 in every block of 5
            indices = torch.arange(num_latents)
            mask = (indices % 5) < 2

            timestep_ids[mask] = history_timestep.item()
            # set prev 1/4 timesteps to history_timestep
        else:
            timestep_ids = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,)).repeat(num_latents)
        timesteps = self.scheduler.timesteps[timestep_ids].to(dtype=self.torch_dtype, device=self.device)
        #Diffusion forcing
        # if "extend" in inputs["input_type"]:
        #     latents_per_exposure = inputs["input_latents"].shape[2] // len(inputs["exposures"])
        #     #remove index 0, 6, 12, 18, etc latents (first frame latents)
        #     idx = torch.arange(inputs["input_latents"].shape[2], device=inputs["input_latents"].device)
        #     mask = (idx % latents_per_exposure != 0)
        #     inputs["input_latents"] = inputs["input_latents"][:, :, mask, :]
        
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timesteps)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timesteps) #doesn't used timesteps nice :) 
        

        noise_pred = self.model_fn(**inputs, timestep=timesteps)
        
        #loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        #loss = loss * self.scheduler.training_weight(timesteps) #haven't updated
        #diffusion forcing loss
        loss_type = "l1"
        if loss_type == "l1":
            loss = torch.abs(noise_pred.float() - training_target.float()).mean(dim=[0,1,3,4])
            loss = torch.mean(loss * self.scheduler.training_weight(timesteps).to(loss.device))
        elif loss_type == "l2":
            loss = (noise_pred.float() - training_target.float()).pow(2).mean(dim=[0,1,3,4])
            loss = torch.mean(loss * self.scheduler.training_weight(timesteps).to(loss.device))
        else:
            exit("loss_type must be either 'l1' or 'l2'")

        return loss


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.merge_decoder is not None:
            dtype = next(iter(self.merge_decoder.parameters())).dtype

            #Blocking for now
            # enable_vram_management(
            #     self.merge_decoder,
            #     module_map = {
            #         torch.nn.Linear: AutoWrappedLinear,
            #         torch.nn.Conv2d: AutoWrappedModule,
            #         torch.nn.SiLU: AutoWrappedModule,
            #     },
            #     module_config = dict(
            #         offload_dtype=dtype,
            #         offload_device="cpu",
            #         onload_dtype=dtype,
            #         onload_device="cpu",
            #         computation_dtype=dtype,
            #         computation_device=self.device,
            #     ),
            # )
      
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                #"models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                #"Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                #"models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
                "Wan2.2_VAE.pth": "Wan-AI/Wan2.2-TI2V-5B",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")
        pipe.audio_encoder = model_manager.fetch_model("wans2v_audio_encoder")
        pipe.merge_decoder = model_manager.fetch_model("wan_video_vae_merge_decoder")
        if pipe.merge_decoder is None:
            pipe.merge_decoder = (
                WanVideoVAEMergeDecoder()
            )
    

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        #set tokenzier config, skip download
        tokenizer_config.skip_download = True
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary(use_usp=use_usp)
            from transformers import Wav2Vec2Processor
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        encoder_decoder_mode = None,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        condition_video: Optional[np.array] = None,
        exposures: Optional[torch.Tensor] = None,
        generate_exposures = (-4, 0, 4),
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        #num_frames=None,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        use_vae_ea: Optional[bool] = False,
    ):

        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }

        from .wan_video_sampler_scheduler import ValScheduler

       
        val_scheduler = ValScheduler(condition_video, generate_exposures, self, encoder_decoder_mode, tiled, tile_size, tile_stride, use_vae_ea=use_vae_ea)

        while True:
            job = val_scheduler.generate_next()
            if job is None:
                break

            inputs_shared = {
                "condition_video": job["video_segment"],
                "video_latents": job.get("video_latents", None),
                "prev_frames_base": job.get("prev_frames_base", None),
                "prev_frames_up": job.get("prev_frames_up", None),
                "prev_frames_down": job.get("prev_frames_down", None),
                "input_type": job["input_type"],
                # "condition_video": condition_video, 
                # "input_type": input_type,
                "encoder_decoder_mode": encoder_decoder_mode,
                "exposures": exposures,
                "end_image": end_image,
                "input_video": input_video, "denoising_strength": denoising_strength,
                "control_video": control_video, "reference_image": reference_image,
                "seed": seed, "rand_device": rand_device,
                "height": height, "width": width, 
                "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
                "sigma_shift": sigma_shift,
                "motion_bucket_id": motion_bucket_id,
                "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
                "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
                "switch_DiT_boundary": switch_DiT_boundary,
                "cfg_scale": cfg_scale,
                "cfg_merge": cfg_merge,
                "test": True
            }
            
            for unit in self.units:
                inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
            
            if inputs_shared["input_type"] == "crf_extend":
                pass
                # TEST FOR EXTENSION MODE
                #out_idx = 1
                #num_frames_per_exposure = len(condition_video)
                # bracket_video_pr = self.preprocess_video(bracket_video)
                # normal_video = bracket_video_pr[:, :, (out_idx+0)*num_frames_per_exposure:(out_idx+1)*num_frames_per_exposure]
                # short_video  = bracket_video_pr[:, :, (out_idx+1)*num_frames_per_exposure:(out_idx+2)*num_frames_per_exposure]
                # long_video   = bracket_video_pr[:, :, (out_idx+2)*num_frames_per_exposure:(out_idx+3)*num_frames_per_exposure]
                # normal_exposure_latents = self.vae.encode(normal_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
                # short_exposure_latents  = self.vae.encode(short_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
                # long_exposure_latents   = self.vae.encode(long_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
 

                # im = (job["prev_frames_base"][0,:,0].cpu().permute(1,2,0)*255).float().numpy().astype(np.uint8)
                # Image.fromarray(im).save(f"debug_prev_frames_base.png")

            # Sample
            inputs_shared_out =self.sample(inputs_shared, inputs_posi, inputs_nega)

            val_scheduler.commit_result(inputs_shared_out["latents"])
        

        #val_scheduler #merge and output
        outputs = val_scheduler.merge_and_output()

        #outputs = self.decode_latents_hdr_merge(inputs_shared_out["latents"], encoder_decoder_mode, inputs_shared_out["exposures"], tiled, tile_size, tile_stride, device=self.device)
        self.load_models_to_device([])

        val_scheduler.clean_up()
        del val_scheduler, inputs_shared_out
        return outputs
    
    def sample(self, inputs_shared, inputs_posi, inputs_nega,):
                # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        # Scheduler
        self.scheduler.set_timesteps(inputs_posi["num_inference_steps"], denoising_strength=inputs_shared["denoising_strength"], shift=inputs_shared["sigma_shift"])

        noise_init = inputs_shared["latents"].clone()

        for progress_id, timestep in enumerate(tqdm(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < inputs_shared["switch_DiT_boundary"] * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            inputs_shared['input_type'] = "crf"
            noise_pred_video = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            inputs_shared["cfg_scale"] = 1.0
            if inputs_shared["cfg_scale"] != 1.0 and inputs_shared["prev_frames_base"] is not None:
                inputs_shared['input_type'] = "crf_extend"
                inputs_shared_modified = inputs_shared.copy()
                latents = self.scheduler.add_noise(inputs_shared["latents"], noise_init, timesteps=self.scheduler.timesteps[0:1].repeat_interleave(noise_init.shape[2], dim=0))
                inputs_shared_modified["prev_frames_base"] = noise_init[0,:, 0:2]
                inputs_shared_modified["prev_frames_up"] = noise_init[0,:, 2:4]
                inputs_shared_modified["prev_frames_down"] = noise_init[0,:, 4:6]
                first_denoising_time = self.scheduler.timesteps[0].unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                noise_pred_no_history = self.model_fn(**models, **inputs_shared_modified, **inputs_posi, timestep=timestep, timestep_history=first_denoising_time)
                noise_pred = noise_pred_no_history + inputs_shared["cfg_scale"] * (noise_pred_video - noise_pred_no_history)
            else:
                noise_pred = noise_pred_video

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
        if inputs_shared["encoder_decoder_mode"] is not None:
            self.load_models_to_device(['merge_decoder'])
        
       # num_frames_per_exposure = num_frames // self.exposures
        num_latents = inputs_shared["latents"].shape[2]
        #assert num_latents is divisible by exposures
        assert num_latents % len(inputs_shared["exposures"]) == 0, f"num_latents {num_latents} must be divisible by exposures {self.exposures}"

        return inputs_shared

    
    # def decode_latents_hdr_merge(self, latents, encoder_decoder_mode, exposures, tiled=True, tile_size=(30, 52), tile_stride=(15, 26), device="cpu"):
    #     num_latents = latents.shape[1]
    #     assert num_latents % len(exposures) == 0, f"num_latents {num_latents} must be divisible by exposures {len(exposures)}"

    #     num_latents_per_exposure = num_latents // len(exposures)
    #     videos = []
    #     for i in range(len(exposures)):
    #         video = self.vae.decode(latents[:,i], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=torch.float32, device=device)
    #         videos.append(self.vae_output_to_video(video, mode="tensor"))

    #     hdr_video = self.merge_decoder(videos, exposures*4, encoder_decoder_mode, mem_efficient=True)
    #     combined_video = torch.cat(videos, dim=2)
    #     outputs = {"hdr_video": hdr_video}

    #     if 'normal_video' in locals():
    #         outputs["normal_video"] = videos[0]
    #     if 'short_video' in locals():
    #         outputs["short_video"] = videos[1]
    #     if 'long_video' in locals():
    #         outputs["long_video"] = videos[2]
    #     if 'combined_video' in locals():
    #         outputs["combined_video"] = combined_video
    #     return outputs

    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1, mode="pil"):
        # To PIL.Image (uint8), NumPy (float [0,1]), Torch (float [0,1]) or same-shape Tensor
        if mode == "tensor":
            return ((vae_output - min_value) / (max_value - min_value)).clamp(0, 1)

        x = vae_output if pattern == "H W C" else reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        x = ((x - min_value) / (max_value - min_value)).clamp(0, 1)

        if mode == "pil":
            x_u8 = (x * 255).to(device="cpu", dtype=torch.uint8)
            return Image.fromarray(x_u8.numpy())
        if mode == "numpy":
            return x.detach().cpu().float().numpy()
        if mode == "torch":
            return x
        raise ValueError('mode must be "pil", "numpy", "torch", or "tensor"')


    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1, mode="pil"):
        # To list[PIL.Image], numpy.ndarray (T,H,W,C), torch.Tensor (T,H,W,C) or same-shape Tensor
        if mode == "tensor":
            return ((vae_output - min_value) / (max_value - min_value)).clamp(0, 1)

        x = vae_output if pattern == "T H W C" else reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")

        frames = [
            self.vae_output_to_image(frame, pattern="H W C", min_value=min_value, max_value=max_value, mode=mode)
            for frame in x
        ]

        if mode == "pil":
            return frames
        if mode == "numpy":
            import numpy as np
            return np.stack(frames, axis=0)
        if mode == "torch":
            return torch.stack(frames, dim=0)
        raise ValueError('mode must be "pil", "numpy", "torch", or "tensor"')


    def custom_check_resize_height_width(self, height, width, num_frames, exposures):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            num_frames_per_exposure = num_frames // len(exposures)
            assert num_frames_per_exposure % self.time_division_factor == self.time_division_remainder, "num_frames_per_exposure must be of the form 4N+1."
            return height, width, num_frames

class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "exposures", "seed", "rand_device", "vace_reference_image", "test"))

    def process(self, pipe, height, width, num_frames, exposures, seed, rand_device, vace_reference_image, test):
        num_frames_per_exposure = num_frames // len(exposures)
        length = ((num_frames_per_exposure - 1) // pipe.time_division_factor + 1)* len(exposures)

        if not test:
            length = length #- len(exposures) #this is cause I remove one frame in the encoding process for the "seperate" and "seperate_debevec" modes (front or back)

        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)

        return {"noise": noise}
    

    
class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("condition_video", "height", "width", "num_frames", "exposures", "test"))

    def process(self, pipe: WanVideoPipeline, condition_video, height, width, num_frames, exposures, test):
        if num_frames is None:
            num_frames = condition_video.shape[0] * 4
        
        exposures = torch.tensor(exposures, dtype=torch.float32, device=pipe.device)
        height, width, num_frames = pipe.custom_check_resize_height_width(height, width, num_frames, exposures)
        return {"height": height, "width": width, "num_frames": num_frames, "exposures": exposures}
    
    
class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "exposures", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image", "encoder_decoder_mode", "input_type"),
            onload_model_names=("vae",)
        )

    #assume input_video is 12
    def process(self, pipe: WanVideoPipeline, input_video, exposures, noise, tiled, tile_size, tile_stride, vace_reference_image, encoder_decoder_mode, input_type):
        if input_video is None:
            return {"latents": noise} #when running inference there is no input video, so this happens
        pipe.load_models_to_device(["vae"])

        num_frames = len(input_video)
        assert num_frames % len(exposures) == 0, f"input_video length must be divisible by exposures={len(exposures)}"
        num_frames_per_exposure = num_frames // len(exposures) #
        # assert 0 == (num_frames_per_exposure-1) % pipe.time_division_factor, "HDR frames must be multiple of 4N+1"

        input_video = pipe.preprocess_video(input_video)

        out_idx = 1
        normal_video = input_video[:, :, (out_idx+0)*num_frames_per_exposure:(out_idx+1)*num_frames_per_exposure]
        short_video  = input_video[:, :, (out_idx+1)*num_frames_per_exposure:(out_idx+2)*num_frames_per_exposure]
        long_video   = input_video[:, :, (out_idx+2)*num_frames_per_exposure:(out_idx+3)*num_frames_per_exposure]

        if encoder_decoder_mode in ["seperate", "seperate_debevec"]:
            drop_index = np.random.choice([0])

            drop = lambda x: x[:, :, drop_index:drop_index+5, ...] 

            #first_latent_num_frames = 0 if drop_index == 0 else 1

            normal_exposure_latents = drop(pipe.vae.encode(normal_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device))
            short_exposure_latents  = drop(pipe.vae.encode(short_video,  device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device))
            long_exposure_latents   = drop(pipe.vae.encode(long_video,   device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device))

            crf_video = input_video[:, :, 0:num_frames_per_exposure]
            crf_latents = drop(pipe.vae.encode(crf_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device))

            input_latents = torch.concat([crf_latents, normal_exposure_latents, short_exposure_latents, long_exposure_latents], dim=2)

        elif encoder_decoder_mode == "seperate_train":
            latent_segments = []
            for i in range(len(exposures)):
                video_segment = input_video[:, :, i*num_frames_per_exposure:(i+1)*num_frames_per_exposure]
                latents_segment = pipe.vae.encode(video_segment, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                latent_segments.append(latents_segment)
            input_latents = torch.stack(latent_segments, dim=1)
        
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            print("this should never be hit I believe")
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}
        
class WanVideoUnit_CustomVideoConditionEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("condition_video", "latents", "height", "width", "tiled", "tile_size", "tile_stride", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, condition_video, latents, height, width, tiled, tile_size, tile_stride, input_latents=None):
        # if condition_video is None or not pipe.dit.fuse_vae_embedding_in_latents: #OLD
        #     return {}

        if condition_video is None: #training case
            return {"latents": latents, "fuse_vae_embedding_in_latents": True} #train time
        else: #testing case
            pipe.load_models_to_device(self.onload_model_names)
            
            condition_video = pipe.preprocess_video(condition_video)
            z = pipe.vae.encode(condition_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

            return {"latents": latents, "fuse_vae_embedding_in_latents": True, "condition_video_latents": z}



class WanVideoUnit_PromptEmbedder(PipelineUnit):

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}






class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}

class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega

