# VAE Exposure Attention — Decoder Changes

## What was added

Cross-exposure attention (EA) inside the VAE decoder, so that when decoding
bracketed-exposure latents the decoder can attend across exposures at the
bottleneck before upsampling.

---

## Files changed

### `diffsynth/models/wan_video_vae_with_ea.py`

Five new classes appended at the bottom (the rest of the file is identical to
`wan_video_vae.py`):

**`ExposureAttentionBlock(dim, num_heads=4)`**
- Self-attention over the exposure dimension.
- Input shape: `(B*E, C, T, H, W)` — all E exposures stacked in the batch dim.
- Reshapes to `(B*T*H*W, E, C)`, runs pre-norm MHA across E, reshapes back.
- Output projection is **zero-initialized** → block is a no-op at init, safe
  for finetuning from a pretrained VAE checkpoint.

**`Decoder3dWithEA`** (subclass of `Decoder3d`)
- For `VideoVAE_`-based models (non-38).
- Same architecture as `Decoder3d` plus one `ExposureAttentionBlock` inserted
  **after the middle blocks** (bottleneck, lowest spatial resolution).
- `forward(x, num_exposures=1, feat_cache=None, feat_idx=[0])`

**`VideoVAEWithEA_`** (subclass of `VideoVAE_`)
- For non-38 VAE models.
- Swaps `self.decoder` from `Decoder3d` to `Decoder3dWithEA`.
- Overrides `decode(z, scale, num_exposures=1)` to pass `num_exposures` down
  through the causal frame-by-frame decode loop.

**`Decoder3d_38WithEA`** (subclass of `Decoder3d_38`)
- For `VideoVAE38_`-based models (Wan2.2 VAE — 12-channel output, `dec_dim=256`).
- Same as `Decoder3d_38` plus `ExposureAttentionBlock` after the middle blocks.
- `forward(x, num_exposures=1, feat_cache=None, feat_idx=[0], first_chunk=False)`
  — matches `Decoder3d_38.forward` signature exactly.

**`VideoVAE38_WithEA`** (subclass of `VideoVAE38_`)
- For the actual Wan2.2 VAE (`VideoVAE38_`, encoder `dim=160`, decoder `dec_dim=256`).
- Swaps `self.decoder` from `Decoder3d_38` to `Decoder3d_38WithEA`.
- Overrides `decode(z, scale, num_exposures=1)` — same as `VideoVAE38_.decode`
  (including the `first_chunk=True` and `unpatchify` call) but passes
  `num_exposures` to the decoder.

---

### `examples/wanvideo/model_training/train_decoder.py`

**`WanDecoderTrainingModule.__init__`** — two new params:
- `use_vae_ea: bool = False` — enables the EA decode path.
- `ea_num_heads: int = 4` — number of heads in `ExposureAttentionBlock`.

When `use_vae_ea=True`, the swap logic:
1. Detects whether `pipe.vae.model` is `VideoVAE38_` (Wan2.2) or `VideoVAE_`.
2. Reads the actual architecture dims from the loaded model:
   - encoder `dim` → `old.dim`
   - decoder `dec_dim` → `old.decoder.dim` (separate from encoder dim in 38 variant)
   - `dim_mult`, `num_res_blocks`, `attn_scales`, `temperal_downsample`
3. Creates `VideoVAE38_WithEA` or `VideoVAEWithEA_` with those exact dims.
4. Copies pretrained weights with `strict=False` — all existing weights transfer,
   only `ea_block` is new (zero-initialized).
5. Replaces `pipe.vae.model` in-place before `switch_pipe_to_training_mode`
   so EA params are included in the trainable set.

**`_decode_with_vae_ea(latents, exposures, tiled)`** — new method:
- Stacks all E exposures' latents along the batch dim: `latents[0]` →
  `(E, C, T', H', W')`.
- Calls `pipe.vae.model.decode(..., num_exposures=E)` directly so EA fires.
- Casts output to `float32` (merge decoder expects float32).
- Converts each decoded exposure to `[0, 1]` via `pipe.vae_output_to_video`.
- Feeds the list of E decoded videos to `pipe.merge_decoder` (same as before).

**`forward`** — routes to `_decode_with_vae_ea` when `self.use_vae_ea=True`,
otherwise falls through to the existing `pipe.decode_latents_hdr_merge`.

---

### `diffsynth/configs/finetune_decoder_justmerger.yaml`

```yaml
use_vae_ea: false      # flip to true to enable EA in the VAE decoder
ea_num_heads: 4        # attention heads (only used when use_vae_ea: true)
                       # when use_vae_ea: true, also set:
                       # trainable_models: "vae.model.decoder.ea_block,merge_decoder"
```

### `diffsynth/configs/finetune_decoder_justmerger_stage2ea.yaml`

Stage 2 config that resumes from the same checkpoint directory as stage 1
with EA enabled and a 10x lower learning rate:

```yaml
use_vae_ea: true
ea_num_heads: 4
learning_rate: 0.00001
trainable_models: "vae.model.decoder.ea_block,merge_decoder"
```

---

## How to enable EA training

In any decoder finetune YAML:

```yaml
use_vae_ea: true
ea_num_heads: 4
trainable_models: "vae.model.decoder.ea_block,merge_decoder"
```

To train the full decoder alongside EA:

```yaml
trainable_models: "vae.model.decoder,merge_decoder"
```

---

## Data flow with EA enabled

```
latents (B, E, C, T', H', W')
  │
  └─ stack exposures into batch → (E, C, T', H', W')
       │
       └─ VideoVAE38_WithEA.decode(num_exposures=E)
            │  conv1 + middle blocks
            │  ExposureAttentionBlock  ← attends across E at bottleneck (dim=1024)
            │  upsamples + head (12-ch) + unpatchify
            └─ (E, C, T, H, W)  [float32]
                 │
                 split → list of E videos (1, C, T, H, W) in [0, 1]
                              │
                              └─ WanVideoVAEMergeDecoder → HDR output
```

---

## Architecture notes

The Wan2.2 VAE (`Wan2.2_VAE.pth`) uses `VideoVAE38_` with:
- Encoder: `Encoder3d_38`, `dim=160`
- Decoder: `Decoder3d_38`, `dec_dim=256` → bottleneck dim = 1024
- z_dim: 48
- Head: 12-channel output (→ `unpatchify` restores to 3-channel RGB)

The EA block sits at the 1024-dim bottleneck with `ea_num_heads=4`.
