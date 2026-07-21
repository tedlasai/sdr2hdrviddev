#!/bin/bash
# EV sensitivity analysis: for ev in (2, 6, 8), train the unet (finetune.sh),
# then the merge decoder (finetune_decoder.sh) for that ev, then run inference
# on both the stuttgart and ubc evaluation sets.
#
# Usage:
#   bash sensitivity_analysis.sh
#
# Runs inside a detached tmux session named "sensitivity_analysis" so it
# survives terminal/SSH disconnects. Attach with:
#   tmux attach -t sensitivity_analysis

set -euo pipefail

SESSION="sensitivity_analysis"
REPO_DIR="/data2/saikiran.tedla/hdrvideo/diff"

if [ "$1" != "--inner" ]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
        exit 1
    fi
    tmux new-session -d -s "$SESSION" "bash '$0' --inner"
    echo "Started tmux session '$SESSION'."
    echo "Attach with: tmux attach -t $SESSION"
    echo "Logs are written to: $REPO_DIR/sensitivity_analysis_logs/"
    exit 0
fi

# ----------------------------------------------------------------------------
# Everything below runs inside the tmux session.
# ----------------------------------------------------------------------------

cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OPENCV_IO_ENABLE_OPENEXR=1
export NCCL_P2P_LEVEL=2
export NCCL_P2P_DISABLE=1
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_BLOCKING_WAIT=0

LOG_DIR="$REPO_DIR/sensitivity_analysis_logs"
mkdir -p "$LOG_DIR"

for ev in 2 6 8; do
    echo "===================================================================="
    echo "EV=${ev}: unet training (finetune.sh)"
    echo "===================================================================="
    bash finetune.sh "$REPO_DIR/diffsynth/configs/threeexposures_crfchanging_${ev}ev.yaml" \
        2>&1 | tee "$LOG_DIR/unet_${ev}ev.log"

    echo "===================================================================="
    echo "EV=${ev}: decoder training (finetune_decoder.sh)"
    echo "===================================================================="
    bash finetune_decoder.sh "$REPO_DIR/diffsynth/configs/finetune_decoder_justmerger_${ev}ev.yaml" \
        2>&1 | tee "$LOG_DIR/decoder_${ev}ev.log"

    echo "===================================================================="
    echo "EV=${ev}: freezing final decoder checkpoint"
    echo "===================================================================="
    DECODER_CKPT_DIR="$REPO_DIR/models/train/finetune_decoder_justmerger_${ev}ev/checkpoints"
    LATEST_DECODER_CKPT=$(ls -t "$DECODER_CKPT_DIR"/epoch-*.safetensors | head -n1)
    if [ -z "$LATEST_DECODER_CKPT" ]; then
        echo "ERROR: no decoder checkpoint found in $DECODER_CKPT_DIR" >&2
        exit 1
    fi
    cp "$LATEST_DECODER_CKPT" "$DECODER_CKPT_DIR/merge_decoder_final.safetensors"
    echo "Copied $LATEST_DECODER_CKPT -> $DECODER_CKPT_DIR/merge_decoder_final.safetensors"

    for dataset in stuttgart ubc; do
        echo "===================================================================="
        echo "EV=${ev}: testing on ${dataset}"
        echo "===================================================================="
        accelerate launch test.py \
            --config "$REPO_DIR/diffsynth/configs/threeexposures_crfchanging_test_val_${ev}ev.yaml" \
            --eval_dataset "$dataset" \
            --output_name "ours${ev}ev" \
            2>&1 | tee "$LOG_DIR/test_${ev}ev_${dataset}.log"
    done
done

echo "===================================================================="
echo "sensitivity_analysis complete for ev = 2, 6, 8"
echo "Outputs: evaluations/ours{2,6,8}ev_{stuttgart,ubc}"
echo "===================================================================="
