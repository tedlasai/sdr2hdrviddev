import torch
from typing import Optional, Union

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models.wan_video_dit import WanModel
from einops import rearrange, repeat, reduce
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d



def model_fn_wan_video(
    dit: WanModel,
    input_type,
    test,
    video_latents=None,
    prev_frames_base = None,
    prev_frames_up = None,
    prev_frames_down = None,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache: bool = None, #I changed this to bool so this works.
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    exposures: Optional[torch.Tensor] = None,
    condition_video_latents: Optional[torch.Tensor] = None,
    input_latents: Optional[torch.Tensor] = None,
    timestep_history: Optional[torch.Tensor] = None,
    first_latent_num_frames: Optional[int] = None,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    assert input_type is not None, "input_type must be specified"
    NUM_LATENTS_PER_EXPOSURE = latents.shape[2] // len(exposures)

    num_frames_known= 2

    def replace_condition_latents(x, condition_video_latents=None, input_latents=None):
        if test: #test
            if input_type == "crf":
                x[:, :, NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1, :] = condition_video_latents
            elif input_type == "nocrf":
                x[:, :, NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1, :] = 0
                x[:, :, NUM_LATENTS_PER_EXPOSURE*1:NUM_LATENTS_PER_EXPOSURE*2, :] = prev_frames_base
            elif input_type == "crf_extend":
                x[:, :, NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1, :] = condition_video_latents
                x[:, :, NUM_LATENTS_PER_EXPOSURE*1:NUM_LATENTS_PER_EXPOSURE*1 + num_frames_known, :] = prev_frames_base
                x[:, :, NUM_LATENTS_PER_EXPOSURE*2:NUM_LATENTS_PER_EXPOSURE*2 + num_frames_known, :] = prev_frames_down
                x[:, :, NUM_LATENTS_PER_EXPOSURE*3:NUM_LATENTS_PER_EXPOSURE*3 + num_frames_known, :] = prev_frames_up
            elif input_type == "nocrf_extend":
                x[:, :, NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1, :] = 0
                x[:, :, NUM_LATENTS_PER_EXPOSURE*1:NUM_LATENTS_PER_EXPOSURE*2, :] = prev_frames_base
                x[:, :, NUM_LATENTS_PER_EXPOSURE*2:NUM_LATENTS_PER_EXPOSURE*2 + num_frames_known, :] = prev_frames_down
                x[:, :, NUM_LATENTS_PER_EXPOSURE*3:NUM_LATENTS_PER_EXPOSURE*3 + num_frames_known, :] = prev_frames_up

            else:
                exit("input_type must be either 'crf' or 'nocrf'")
        else: #train
            input_latents = input_latents.detach() #make sure no gradients flow through input latents
            if input_type == "crf":
                #subttract
                x[:, :, NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1, :] = input_latents[:,:, NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1, :]
            elif input_type == "nocrf":
                x[:, :, NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1, :] = 0
                x[:, :, NUM_LATENTS_PER_EXPOSURE*1:NUM_LATENTS_PER_EXPOSURE*2, :] = input_latents[:,:, NUM_LATENTS_PER_EXPOSURE*1:NUM_LATENTS_PER_EXPOSURE*2, :]
            elif input_type == "crf_extend":
                x[:, :, NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1, :] = input_latents[:,:, NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1, :]
                x[:, :, NUM_LATENTS_PER_EXPOSURE*1:NUM_LATENTS_PER_EXPOSURE*1 + num_frames_known, :] = input_latents[:,:, NUM_LATENTS_PER_EXPOSURE*1:NUM_LATENTS_PER_EXPOSURE*1+num_frames_known, :]
                x[:, :, NUM_LATENTS_PER_EXPOSURE*2:NUM_LATENTS_PER_EXPOSURE*2 + num_frames_known, :] = input_latents[:,:, NUM_LATENTS_PER_EXPOSURE*2:NUM_LATENTS_PER_EXPOSURE*2+num_frames_known, :]
                x[:, :, NUM_LATENTS_PER_EXPOSURE*3:NUM_LATENTS_PER_EXPOSURE*3 + num_frames_known, :] = input_latents[:,:, NUM_LATENTS_PER_EXPOSURE*3:NUM_LATENTS_PER_EXPOSURE*3+num_frames_known, :]
            elif input_type == "nocrf_extend":
                x[:, :, NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1, :] = 0
                x[:, :, NUM_LATENTS_PER_EXPOSURE*1:NUM_LATENTS_PER_EXPOSURE*2, :] = input_latents[:,:, NUM_LATENTS_PER_EXPOSURE*1:NUM_LATENTS_PER_EXPOSURE*2, :]
                x[:, :, NUM_LATENTS_PER_EXPOSURE*2:NUM_LATENTS_PER_EXPOSURE*2 + num_frames_known, :] = input_latents[:,:, NUM_LATENTS_PER_EXPOSURE*2:NUM_LATENTS_PER_EXPOSURE*2+num_frames_known, :]
                x[:, :, NUM_LATENTS_PER_EXPOSURE*3:NUM_LATENTS_PER_EXPOSURE*3 + num_frames_known, :] = input_latents[:,:, NUM_LATENTS_PER_EXPOSURE*3:NUM_LATENTS_PER_EXPOSURE*3+num_frames_known, :]
            else:
                exit("input_type must be either 'crf' or 'nocrf'")
        return x
        

    def mask_timestep(mask):
        if input_type == "crf":
            mask[NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1] = 0
        elif input_type == "nocrf":
            #mask[NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1] = 0
            mask[NUM_LATENTS_PER_EXPOSURE*1:NUM_LATENTS_PER_EXPOSURE*2] = 0
        elif input_type == "crf_extend":
            mask[NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1] = 0
            mask[NUM_LATENTS_PER_EXPOSURE*1:NUM_LATENTS_PER_EXPOSURE*1 + num_frames_known] = timestep_history
            mask[NUM_LATENTS_PER_EXPOSURE*2:NUM_LATENTS_PER_EXPOSURE*2 + num_frames_known] = timestep_history
            mask[NUM_LATENTS_PER_EXPOSURE*3:NUM_LATENTS_PER_EXPOSURE*3 + num_frames_known] = timestep_history
        elif input_type == "nocrf_extend":
            #mask[NUM_LATENTS_PER_EXPOSURE*0:NUM_LATENTS_PER_EXPOSURE*1] = 0
            mask[NUM_LATENTS_PER_EXPOSURE*1:NUM_LATENTS_PER_EXPOSURE*2] = 0
            mask[NUM_LATENTS_PER_EXPOSURE*2:NUM_LATENTS_PER_EXPOSURE*2 + num_frames_known] = 0
            mask[NUM_LATENTS_PER_EXPOSURE*3:NUM_LATENTS_PER_EXPOSURE*3 + num_frames_known] = 0
        else:
            exit("input_type must be either 'crf' or 'nocrf'")
        
        return mask

    if video_latents is not None:
        condition_video_latents = video_latents


    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.ones((latents.shape[2], latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep.unsqueeze(1)
        timestep = mask_timestep(timestep).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))


    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x = replace_condition_latents(x, condition_video_latents, input_latents) 

    x, (f, h, w) = dit.patchify(x, None)

    #Is Condition?
    # condition = torch.ones((latents.shape[2], latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) 
    # condition = mask_timestep(condition).flatten()
    # condition_e = dit.condition_embedding(condition.unsqueeze(1))
    # condition_proj = dit.condition_projection(condition_e).unsqueeze(0)
    # x = x + condition_proj
    
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)


    is_crf = torch.cat([torch.ones(NUM_LATENTS_PER_EXPOSURE), torch.zeros(f - NUM_LATENTS_PER_EXPOSURE)], dim=0).to(x.device)-0.5
    exposure_info = exposures.repeat_interleave(NUM_LATENTS_PER_EXPOSURE)
    relative_frame_idx = (torch.arange(len(exposures))/len(exposures)).repeat(NUM_LATENTS_PER_EXPOSURE).to(x.device) - 0.5
    

    all_info_freq_dim = 16
    all_info = torch.cat([sinusoidal_embedding_1d(all_info_freq_dim, is_crf), sinusoidal_embedding_1d(all_info_freq_dim, exposure_info), sinusoidal_embedding_1d(all_info_freq_dim, relative_frame_idx)], dim=-1)
    all_info = all_info.view(f,1,1,-1).expand(f, h, w, -1).reshape(f * h * w, 1, -1)

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, all_info,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs, all_info,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs, all_info)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        print("Shouldn't be in here either")
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))

    x = replace_condition_latents(x, condition_video_latents, input_latents) 
    return x




class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



##### BELOW THIS IS NEVER USED #####
