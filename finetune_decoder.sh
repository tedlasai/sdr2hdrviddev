#!/bin/bash

# Environment variables
export PYTHONPATH="/data2/saikiran.tedla/hdrvideo/diff:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=7
export OPENCV_IO_ENABLE_OPENEXR=1

# Training command
accelerate launch examples/wanvideo/model_training/train_decoder.py --config /data2/saikiran.tedla/hdrvideo/diff/diffsynth/configs/finetune_decoder_justmerger_stage2ea.yaml