#!/bin/bash
# Run script for RunPod environment
# Target: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

set -e

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Ensure we're using Python 3.11
PYTHON_CMD="python3"

echo "=========================================="
echo "4DGS RunPod Execution Script"
echo "Environment: Ubuntu 22.04"
echo "Python: $($PYTHON_CMD --version)"
echo "=========================================="
echo ""

# Check if first argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: bash run_runpod.sh <script> [arguments]"
    echo ""
    echo "Examples:"
    echo "  bash run_runpod.sh tools/preprocess_video.py /videos -o /data"
    echo "  bash run_runpod.sh tools/train.py --data_root /data --out_dir /model"
    echo "  bash run_runpod.sh tools/render.py --ckpt /model/final.pt --out_dir /renders"
    exit 1
fi

# Run the command
echo "Running: $PYTHON_CMD $@"
echo ""
$PYTHON_CMD "$@"