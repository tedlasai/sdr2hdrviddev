import imageio, os, torch, warnings, torchvision, argparse, json
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from einops import rearrange
import sys
import wandb
import torch.distributed as dist
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils import output_frames, output_ldr_video



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)

        def named_modules_recursive(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, child
                # recurse into the child
                yield from named_modules_recursive(child, prefix=full_name)
        
        def freeze_except_recursive(model, model_names):
            #freeze all first
            for name, child in named_modules_recursive(model):
                child.eval()
                child.requires_grad_(False)
            #then unfreeze the specified ones
            for name, model in named_modules_recursive(model):
                if name in model_names:
                    model.train()
                    model.requires_grad_(True)
                    print("Setting to training mode:", name)

        #special code for decoder
        if "vae" in trainable_models:
            freeze_except_recursive(pipe, trainable_models.split(","))
        else:
            # Freeze untrainable models
            pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                #replace "prefix" of lora_base_model + "." from keys in load_result[1] with "" dit.x to x for example
                state_dict = {k.replace(lora_base_model + ".", "", 1): v for k, v in state_dict.items()}

                load_result = model.load_state_dict(state_dict, strict=False)

  
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.best_combined_psnr = float("-inf")

    def _load_existing_best_psnr(self):
        if not os.path.isdir(self.output_path):
            return
        import re
        best_pattern = re.compile(r"^best_(\d+)-(\d+)\.safetensors$")
        best_found = float("-inf")
        for file_name in os.listdir(self.output_path):
            match = best_pattern.match(file_name)
            if match is None:
                continue
            int_part, frac_part = match.groups()
            parsed_psnr = float(f"{int_part}.{frac_part}")
            if parsed_psnr > best_found:
                best_found = parsed_psnr
        if best_found > float("-inf"):
            self.best_combined_psnr = best_found
            print(f"Initialized best combined PSNR from checkpoints: {self.best_combined_psnr:.2f}")




    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator, model, optimizer, scheduler, epoch_id, val_combined_psnr=None):
        # this creates a warning cause it wraps torch.barrier and this requires device ids
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            # ---- prepare state dicts ----
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict,
                remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)

            # ---- paths ----
            os.makedirs(self.output_path, exist_ok=True)
            model_path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            opt_path   = os.path.join(self.output_path, f"epoch-{epoch_id}.optimizer_scheduler.pt")

            print("SAVING")
            
            # ---- save model & optimizer ----
            accelerator.save(state_dict, model_path, safe_serialization=True)
            accelerator.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, opt_path)

            # ---- keep only last 2 checkpoints (model + matching optimizer) ----
            model_files = sorted(
                [f for f in os.listdir(self.output_path)
                if f.startswith("epoch-") and f.endswith(".safetensors")],
                key=lambda x: int(x.split("-")[1].split(".")[0])
            )

            for old_f in model_files[:-2]:
                # remove model checkpoint
                try:
                    os.remove(os.path.join(self.output_path, old_f))
                except FileNotFoundError:
                    pass
                # remove matching optimizer checkpoint
                old_opt = old_f.replace(".safetensors", ".optimizer_scheduler.pt")
                try:
                    os.remove(os.path.join(self.output_path, old_opt))
                except FileNotFoundError:
                    pass

            # ---- save/replace best checkpoint by combined PSNR ----
            
            if val_combined_psnr is not None and val_combined_psnr > self.best_combined_psnr:
                self.best_combined_psnr = val_combined_psnr
                best_tag = f"{val_combined_psnr:.2f}".replace(".", "-")
                best_model_path = os.path.join(self.output_path, f"best_{best_tag}.safetensors")
                best_opt_path = os.path.join(self.output_path, f"best_{best_tag}.optimizer_scheduler.pt")

                # Remove previous best files so the new best replaces them.
                for f in os.listdir(self.output_path):
                    if f.startswith("best_"):
                        try:
                            os.remove(os.path.join(self.output_path, f))
                        except FileNotFoundError:
                            pass

                accelerator.save(state_dict, best_model_path, safe_serialization=True)
                accelerator.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, best_opt_path)
                print(f"New best checkpoint saved: {os.path.basename(best_model_path)} (combined PSNR={val_combined_psnr:.2f})")

        # ensure all processes move on after main process saves/cleans
        accelerator.wait_for_everyone()

    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)

def validate(val_dataloader, model, model_logger, epoch_id, accelerator, dataset, args):
    skip_val = args.skip_val #debugging thing
    max_value = 16.0
    out_metrics,data_gathered,outputs_gathered = None, None, None
    saves = 0
    combined_psnr = None
    val_gpus = 5    
    val_ranks = list(range(min(val_gpus, accelerator.num_processes)))
    is_val_rank = accelerator.process_index in val_ranks
    subgroup_root_rank = val_ranks[0] if len(val_ranks) > 0 else 0
    use_subgroup = dist.is_available() and dist.is_initialized() and len(val_ranks) > 1
    val_group = dist.new_group(ranks=val_ranks) if use_subgroup else None

    accelerator.wait_for_everyone()
    for step, data in enumerate(tqdm(val_dataloader, desc="Validation", disable=not (accelerator.is_local_main_process and is_val_rank))):
        if skip_val:
            break
        if not is_val_rank:
            continue
        with accelerator.accumulate(model):
            if dataset.load_from_cache:
                loss = model({}, inputs=data)
            else:
                condition_video = data["input_video"]
                exposures = data["exposures"]
                unwrapped_model = accelerator.unwrap_model(model)
                import numpy as np
                outputs = unwrapped_model.pipe(
                    prompt=data["prompt"],
                    #negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    condition_video=condition_video, 
                    exposures=exposures,
                    height = args.height,
                    width = args.width,
                    num_inference_steps=50,
                    seed=1, tiled=False,
                    cfg_scale=1.0,
                    encoder_decoder_mode=unwrapped_model.encoder_decoder_mode,
                )
                unwrapped_model.pipe.scheduler.set_timesteps(1000, training=True) #reset timesteps after inference


        from utils import get_psnr_fn, get_ssim_fn, get_lpips_fn, compute_all_metrics, average_metrics, flatten_dict

        
        outputs["dummy_string"] = "dummy" #add string to outputs to make it match data (this makes gathering same across both)
        torch.cuda.empty_cache() #try to free up memory before gathering, this is crucial to avoid OOMs during validation
        if use_subgroup:
            gathered_data = [None for _ in val_ranks]
            gathered_outputs = [None for _ in val_ranks]
            outputs_for_gather = {
                key: (value.detach().cpu() if torch.is_tensor(value) else value)
                for key, value in outputs.items()
            }
            dist.all_gather_object(gathered_data, data, group=val_group)
            dist.all_gather_object(gathered_outputs, outputs_for_gather, group=val_group)
            data_gathered = gathered_data
            outputs_gathered = gathered_outputs
        else:
            data_gathered = [data]
            outputs_gathered = [outputs]
        

        #data, outputs = accelerator.gather_for_metrics((data, outputs), use_gather_object=True)
        if accelerator.process_index == subgroup_root_rank:
            for i in range(len(data_gathered)):
                device = outputs_gathered[0]["hdr_video"].device

                if out_metrics is None:
                    psnr_fn = get_psnr_fn(device)

                    metrics = {
                        "low":  {"psnr": psnr_fn},# "ssim": ssim_fn},#, "lpips": lpips_fn},
                        "mid":  {"psnr": psnr_fn},# "ssim": ssim_fn},#, "lpips": lpips_fn},
                        "high": {"psnr": psnr_fn},# "ssim": ssim_fn},#, "lpips": lpips_fn},
                        "hdr":  {"psnr": psnr_fn},# "ssim": ssim_fn},#, "lpips": lpips_fn},
                        "combined":  {"psnr": psnr_fn},# "ssim": ssim_fn},
                    }
                    out_metrics = {
                    t: {m: torch.empty(0, device=device) for m in ["psnr"]}
                    for t in ["low", "mid", "high", "hdr", "combined"]
                }
                    
                data = data_gathered[i]
                outputs = outputs_gathered[i]
                out_dir = model_logger.output_path / "../" / "val_videos"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                out_hdr_video = outputs["hdr_video"].to(device)
                out_combined_video = outputs["combined_video"].to(device)
                frames_per_exposure = data["bracket_video"].shape[0] // data["exposures"].shape[0]
                in_crf_video = torch.from_numpy(data["bracket_video"][0*frames_per_exposure:1*frames_per_exposure]).unsqueeze(0).to(device)/255 #This scale feels funky here
                gt_hdr_video = torch.from_numpy(data["hdr_video"]).unsqueeze(0).to(device)
                gt_combined_video = torch.from_numpy(data["bracket_video"]).unsqueeze(0).to(device)/255 #This scale feels funky here
                in_crf_video = rearrange(in_crf_video, 'b t h w c -> b t c h w')
                gt_hdr_video = rearrange(gt_hdr_video, 'b t h w c -> b t c h w') 
                gt_combined_video = rearrange(gt_combined_video, 'b t h w c -> b t c h w')
                out_hdr_video = rearrange(out_hdr_video, 'b c t h w -> b t c h w')
                out_combined_video = rearrange(out_combined_video, 'b c t h w -> b t c h w')
                normalized_out_hdr_video = out_hdr_video.to(torch.float32)/max_value
                normalized_gt_hdr_video = gt_hdr_video.to(torch.float32)/max_value

                from utils import generate_multi_exposure_video
                out_multi_exposure_video = generate_multi_exposure_video(normalized_out_hdr_video*max_value, exposures=(-4, 0, 4))
                gt_multi_exposure_video = generate_multi_exposure_video(normalized_gt_hdr_video*max_value, exposures=(-4, 0, 4))

                # ---- delete all outputs from last epoch on disk ----
                if epoch_id >= 1:
                    import shutil, glob, os
                    prev_epoch = epoch_id
                    for f in glob.glob(str(out_dir / f"*epoch-{prev_epoch}_item-*.mp4")):
                        try: os.remove(f)
                        except FileNotFoundError: pass
                    for root in ["low_gt","mid_gt","high_gt","combined_gt","low_pred","mid_pred","high_pred","combined_pred","hdrnorm_gt","hdrnorm_pred","hdr_gt","hdr_pred"]:
                        for p in glob.glob(str(out_dir / root / f"*epoch-{prev_epoch}_item-*")):
                            shutil.rmtree(p, ignore_errors=True)

                # ---- exposure map ----
                exposures = [("low", 0), ("mid", 1), ("high", 2)]
                save_ldr = lambda tensor, rel: output_ldr_video(tensor, str(out_dir / rel), channel_order="NCHW")

                for b in range(normalized_out_hdr_video.shape[0]):

                    if gt_multi_exposure_video[b].shape != out_multi_exposure_video[b].shape:
                        [compute_all_metrics(gt_multi_exposure_video[b, idx], out_multi_exposure_video[b, idx,:gt_multi_exposure_video[b, idx].shape[0]], name, metrics, out_metrics) for name, idx in exposures]
                        [compute_all_metrics(normalized_gt_hdr_video[b], normalized_out_hdr_video[b,:normalized_gt_hdr_video[b].shape[0]], "hdr", metrics, out_metrics)]
                    else:
                        # ----- metrics (now fully one-liner) -----
                        [compute_all_metrics(gt_multi_exposure_video[b, idx], out_multi_exposure_video[b, idx], name, metrics, out_metrics) for name, idx in exposures] + \
                        [compute_all_metrics(normalized_gt_hdr_video[b], normalized_out_hdr_video[b], "hdr", metrics, out_metrics)] + \
                        [compute_all_metrics(gt_combined_video[b][gt_combined_video.shape[1]//4:], out_combined_video[b], "combined", metrics, out_metrics)]
            



                    # ----- videos (fully one-liner too) -----
                    [save_ldr(gt_multi_exposure_video[b, idx], f"{name}_orig_epoch-{epoch_id+1}_item-{saves}.mp4") for name, idx in exposures] 
                    [save_ldr(gt_combined_video[b], f"gt_combined_epoch-{epoch_id+1}_item-{saves}.mp4")] 
                    [save_ldr(out_multi_exposure_video[b, idx], f"{name}_pred_epoch-{epoch_id+1}_item-{saves}.mp4") for name, idx in exposures] 
                    save_ldr(out_combined_video[b], f"out_combined_epoch-{epoch_id+1}_item-{saves}.mp4")
                    save_ldr(in_crf_video[b], f"input_crf_epoch-{epoch_id+1}_item-{saves}.mp4")
                    save_ldr(normalized_gt_hdr_video[b],  f"hdrnorm_gt_epoch-{epoch_id+1}_item-{saves}.mp4")
                    save_ldr(normalized_out_hdr_video[b], f"hdrnorm_pred_epoch-{epoch_id+1}_item-{saves}.mp4")

                    # ----- frames (kept readable) -----
                    for name, idx in exposures:
                        output_frames(gt_multi_exposure_video[b, idx],  str(out_dir / f"{name}_gt" / f"gt_epoch-{epoch_id+1}_item-{saves}"),   mode="ldr", channel_order="NCHW")
                    output_frames(gt_combined_video[b],                 str(out_dir / "combined_gt"  / f"gt_epoch-{epoch_id+1}_item-{saves}"),  mode="ldr", channel_order="NCHW")
                    for name, idx in exposures:
                        output_frames(out_multi_exposure_video[b, idx], str(out_dir / f"{name}_pred"/ f"ours_epoch-{epoch_id+1}_item-{saves}"), mode="ldr", channel_order="NCHW")
                    output_frames(out_combined_video[b],                str(out_dir / "combined_pred"/ f"ours_epoch-{epoch_id+1}_item-{saves}"), mode="ldr", channel_order="NCHW")
                    output_frames(normalized_gt_hdr_video[b],           str(out_dir / "hdrnorm_gt"   / f"gt_epoch-{epoch_id+1}_item-{saves}"),  mode="ldr", channel_order="NCHW")
                    output_frames(normalized_out_hdr_video[b],          str(out_dir / "hdrnorm_pred" / f"ours_epoch-{epoch_id+1}_item-{saves}"),mode="ldr", channel_order="NCHW")
                    output_frames(normalized_gt_hdr_video[b],           str(out_dir / "hdr_gt"       / f"gt_epoch-{epoch_id+1}_item-{saves}"),  mode="hdr", channel_order="NCHW")
                    output_frames(normalized_out_hdr_video[b],          str(out_dir / "hdr_pred"     / f"ours_epoch-{epoch_id+1}_item-{saves}"),mode="hdr", channel_order="NCHW")
                    output_frames(in_crf_video[b],                     str(out_dir / "input_crf"    / f"input_epoch-{epoch_id+1}_item-{saves}"), mode="ldr", channel_order="NCHW")

                    saves += 1
                del out_multi_exposure_video, gt_multi_exposure_video, normalized_out_hdr_video, normalized_gt_hdr_video, gt_combined_video, out_combined_video, in_crf_video
                torch.cuda.empty_cache() #try to free up memory after saving each item, this is crucial to avoid OOMs during validation

                #delete all outputs from last epoch on disk
    if accelerator.process_index == subgroup_root_rank and not skip_val:
        averaged_metrics = average_metrics(out_metrics)
        flattened_metrics = flatten_dict(averaged_metrics)
        accelerator.log(flattened_metrics)
        combined_metric = averaged_metrics.get("combined", {}).get("psnr", None)
        if combined_metric is not None:
            combined_psnr = combined_metric.item() if torch.is_tensor(combined_metric) else float(combined_metric)
    
    
    
    
    del data_gathered, outputs_gathered, out_metrics #crucial to free up memory
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    return combined_psnr


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        epochs_done = args.epochs_done
    

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers, drop_last=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        log_with="wandb",

    )
        
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)

    accelerator.init_trackers(
        project_name="hdrgen",
        init_kwargs={"wandb": {
            "name": "unetfinetune",
        }}
    )

    model, optimizer, dataloader, val_dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, val_dataloader, scheduler)
    from accelerate.utils import broadcast_object_list

    if len(args.optimizer_paths) > 0:
        opt_ckpt = args.optimizer_paths[0]
        opt_state = torch.load(opt_ckpt, map_location="cpu")
        # now every rank has the same dict

        # opt_state = {"optimizer": ..., "scheduler": ...}
        optimizer.load_state_dict(opt_state["optimizer"])
        scheduler.load_state_dict(opt_state["scheduler"])

        accelerator.print(f"Loaded optimizer & scheduler state from: {opt_ckpt}")
        accelerator.wait_for_everyone()
    model_logger._load_existing_best_psnr()
    for epoch_id in range(epochs_done, num_epochs):
        val_combined_psnr = None
        if epoch_id == epochs_done:
            with torch.no_grad():
                val_combined_psnr = validate(val_dataloader, model, model_logger, epoch_id, accelerator, val_dataset, args)
            if save_steps is None:
                model_logger.on_epoch_end(accelerator, model, optimizer, scheduler, epoch_id, val_combined_psnr=val_combined_psnr)
 
        
        for data in tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}"):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                loss = loss.mean()
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()

                # only log on the step that actually syncs grads
                if accelerator.sync_gradients:
                    # Reduce total loss
                    loss_mean = accelerator.reduce(loss.detach(), reduction="mean")
                    if accelerator.is_main_process:
                        accelerator.log({"train/loss": loss_mean})

        #clean up memory
        del loss, loss_mean, data
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        with torch.no_grad():
            val_combined_psnr = validate(val_dataloader, model, model_logger, epoch_id, accelerator, val_dataset, args)


        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, optimizer, scheduler, epoch_id, val_combined_psnr=val_combined_psnr)
 
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers )
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in tqdm(enumerate(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data, return_inputs=True)
                torch.save(data, save_path)



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--config", type=str, default="", required=True, help="Path to the config file.")
    return parser

