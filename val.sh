#!/bin/bash

# Environment variables
export PYTHONPATH="/data2/saikiran.tedla/hdrvideo/diff:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=7
export OPENCV_IO_ENABLE_OPENEXR=1
export NCCL_P2P_LEVEL=2
export NCCL_P2P_DISABLE=1
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_BLOCKING_WAIT=0

# Training command
accelerate launch examples/wanvideo/model_training/train.py --config /data2/saikiran.tedla/hdrvideo/diff/diffsynth/configs/threeexposures_crffixed_test_val.yaml