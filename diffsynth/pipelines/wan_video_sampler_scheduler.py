from collections import deque
import numpy as np
import torch

NEXT_CHUNK = 3 # latent frames
TEMPORAL_STRIDE = 4

def frames_to_latent_idx(f): return 0 if f == 0 else ((f - 1) // TEMPORAL_STRIDE) + 1
def latent_to_frame_idx(l):  return 0 if l == 0 else 1 + (l - 1) * TEMPORAL_STRIDE

def hdr_pixel_frames_to_latent_count(num_hdr_frames: int) -> int:
    """Map HDR pixel-frame count (4n+1 style) to Wan VAE latent-frame count."""
    return frames_to_latent_idx(num_hdr_frames - 1) + 1


class ValScheduler:
    def __init__(self, condition_video, exposures, pipe, encoder_decoder_mode, tiled, tile_size, tile_stride, use_vae_ea=False, num_hdr_frames=17, exp_gap=7, predict_gamma=True):

        self.num_hdr_frames = num_hdr_frames  # pixel frames in the conditioning stream
        self.first_chunk_latents = hdr_pixel_frames_to_latent_count(num_hdr_frames)
        self.special_padding = False

        if self.special_padding:
            full_padded_video, self.pad_idx = [], []
            i, T = 0, condition_video.shape[0]

            while i < T:
                j = min(i + (self.num_hdr_frames if i == 0 else 8), T)
                full_padded_video += list(condition_video[i:j])
                full_padded_video += [condition_video[j-1]] * 4
                self.pad_idx += list(range(len(full_padded_video)-4, len(full_padded_video)))
                i = j

            full_padded_video = np.stack(full_padded_video)
            condition_video = full_padded_video

        num_frames = len(condition_video)
        num_latents = frames_to_latent_idx(num_frames - 1) + 1

        # ===================== PAD VIDEO TO MATCH LATENT SCHEDULE =====================
        num_latents = max(num_latents, self.first_chunk_latents)
        num_latents = self.first_chunk_latents + ((num_latents - self.first_chunk_latents + NEXT_CHUNK - 1) // NEXT_CHUNK) * NEXT_CHUNK

        target_frames = TEMPORAL_STRIDE * (num_latents - 1) + 1

        self.num_end_pad_frames = 0
        num_frames = condition_video.shape[0]

        if num_frames < target_frames:
            self.num_end_pad_frames = target_frames - num_frames
            pad = np.repeat(condition_video[-1:], self.num_end_pad_frames, axis=0)
            condition_video = np.concatenate([condition_video, pad], axis=0)

        num_frames = condition_video.shape[0]
        # ==============================================================================

        self.condition_video = condition_video
        self.exposures = exposures
        self.sorted_exposures = sorted(exposures)
        self.num_frames = num_frames
        self.num_latents = num_latents
        self.condition_latents = None

        self.videos = None
        self.latents = None
        self.out_frames = None
        # Use the first (lowest) exposure as the scheduling sentinel
        self._sentinel = self.sorted_exposures[0]
        self.done_latents = {e: 0 for e in exposures}
        self.instruction = None

        self.pipe = pipe
        self.encoder_decoder_mode = encoder_decoder_mode
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.use_vae_ea = use_vae_ea
        self.exp_gap = int(exp_gap)
        self.predict_gamma = predict_gamma

        self.latent_condition = True

    def exposure_complete(self):
        return self.done_latents[self._sentinel] >= self.num_latents

    def next_chunk_size_latents(self):
        if self.done_latents[self._sentinel] == 0:
            return self.first_chunk_latents
        return NEXT_CHUNK

    def generate_next(self):
        if self.exposure_complete():
            return None  # nothing left to do

        l_start = self.done_latents[self._sentinel]
        l_end = min(self.num_latents, l_start + self.next_chunk_size_latents())

        f_start = latent_to_frame_idx(l_start)
        f_end = latent_to_frame_idx(l_end)

        if self.latent_condition:
            if self.condition_latents is None:
                encoded = self.pipe.vae.encode(
                    self.pipe.preprocess_video(self.condition_video),
                    device=self.pipe.device,
                    tiled=self.tiled,
                    tile_size=self.tile_size,
                    tile_stride=self.tile_stride,
                ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                self.condition_latents = encoded
            video_latents = self.condition_latents[:, :, l_start:l_end]
        else:
            video_latents = None

        self.instruction = {
            "video_segment": self.condition_video[f_start:f_end],
            "video_latents": video_latents,
            "cond_exposure": 0,
            "generating_exposures": self.exposures,
            "f_start": f_start,
            "f_end": f_end,
            "l_start": l_start,
            "l_end": l_end,
            "input_type": "crf",
            "prev_frames_base": None,
            "prev_frames_up": None,
            "prev_frames_down": None,
        }
        return self.instruction

    def commit_result(self, new_latents):
        num_latents_per_exposure = new_latents.shape[2] // 4
        # Model always outputs 4 channels: [CRF, low, base, high]
        # Map directly to sorted exposures: sorted_exps[0]=low, [1]=base, [2]=high
        low_video  = new_latents[:,:,1*num_latents_per_exposure:2*num_latents_per_exposure]
        base_video = new_latents[:,:,2*num_latents_per_exposure:3*num_latents_per_exposure]
        high_video = new_latents[:,:,3*num_latents_per_exposure:4*num_latents_per_exposure]

        self.create_latents(new_latents.shape[1:], new_latents.device, new_latents.dtype)
        self.create_videos(self.condition_video.shape, self.exposures, new_latents.device, new_latents.dtype)

        l_start = self.instruction["l_start"]
        l_end   = self.instruction["l_end"]
        f_start = self.instruction["f_start"]
        f_end   = self.instruction["f_end"]
        chunk_l = l_end - l_start
        chunk_f = f_end - f_start

        for exp, vid in zip(self.sorted_exposures, [low_video, base_video, high_video]):
            self.latents[exp][:, l_start:l_end] = vid[:, :, -chunk_l:]
            self.done_latents[exp] = l_end
            self.videos[exp][:, :, f_start:f_end] = self.decode_latents(vid)[:, :, -chunk_f:]

    def create_latents(self, latents_shape, device, dtype):
        if self.latents is None:
            print(f"Creating latents with shape {latents_shape} for exposures {self.exposures}")
            self.latents = {e: torch.zeros((latents_shape[0], self.num_latents, latents_shape[2], latents_shape[3]), device=device, dtype=dtype) for e in self.exposures}

    def create_videos(self, video_shape, exposures, device, dtype):
        if self.videos is None:
            self.videos = {}
            for e in exposures:
                self.videos[e] = torch.zeros((1, video_shape[3], self.num_frames, video_shape[1], video_shape[2]), device=device, dtype=dtype)

    def decode_latents(self, latents):
        video = self.pipe.vae.decode(latents, device=latents.device, tiled=self.tiled, tile_size=self.tile_size, tile_stride=self.tile_stride).to(dtype=torch.float32, device=latents.device)
        video = self.pipe.vae_output_to_video(video, mode="tensor")
        print(f"Video min: {video.min()}, Video max: {video.max()}")
        return video

    def _decode_latents_ea(self):
        sorted_exposures = self.sorted_exposures
        E = len(sorted_exposures)
        all_latents = torch.stack([self.latents[e] for e in sorted_exposures], dim=0)  # (E, C, T', H', W')
        vae_device = next(self.pipe.vae.model.parameters()).device
        all_latents = all_latents.to(vae_device)
        with torch.no_grad():
            raw = self.pipe.vae.model.decode(all_latents, self.pipe.vae.scale, num_exposures=E, tiled=self.tiled)
        raw = raw.clamp_(-1, 1).to(dtype=torch.float32)
        return {e: self.pipe.vae_output_to_video(raw[i:i+1], mode="tensor") for i, e in enumerate(sorted_exposures)}

    def merge_and_output(self):
        if self.latent_condition:
            if self.use_vae_ea:
                self.videos = self._decode_latents_ea()
            else:
                self.videos = {e: self.decode_latents(self.latents[e].unsqueeze(0)) for e in self.exposures}
        else:
            self.videos = {e: self.videos[e].to(dtype=torch.float32) for e in self.exposures}

        videos_tensor = torch.stack([self.videos[e] for e in self.sorted_exposures], dim=1)
        exposures = torch.tensor([e for e in self.sorted_exposures], device=videos_tensor.device, dtype=videos_tensor.dtype)

        from utils import output_ldr_video
        for i, e in enumerate(self.sorted_exposures):
            output_ldr_video(videos_tensor[0,i].permute(1,2,3,0), f"debug_video_{e}.mp4", fps=30)

        if self.num_end_pad_frames > 0:
            videos_tensor = videos_tensor[:, :, :, :-self.num_end_pad_frames]

        if self.special_padding:
            keep = torch.ones(videos_tensor.shape[3], dtype=torch.bool, device=videos_tensor.device)
            keep[self.pad_idx] = False
            videos_tensor = videos_tensor[:, :, :, keep]

        print("Merging videos with encoder-decoder mode:", self.encoder_decoder_mode)
        hdr_video = self.pipe.merge_decoder(videos_tensor, exposures, self.encoder_decoder_mode, mem_efficient=True, predict_gamma=self.predict_gamma)
        combined_video = torch.cat([self.videos[self.sorted_exposures[0]], self.videos[self.sorted_exposures[1]], self.videos[self.sorted_exposures[2]]], dim=2)
        torch.cuda.empty_cache()

        return {
            "combined_video": combined_video,
            "hdr_video": hdr_video,
        }

    def clean_up(self):
        del self.condition_video
        del self.condition_latents
        del self.latents
        del self.instruction
        del self.videos
        torch.cuda.empty_cache()
