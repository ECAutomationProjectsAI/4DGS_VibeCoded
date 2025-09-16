#!/bin/bash
# 4D Gaussian Splatting Installation Script
# Optimized for RunPod and similar GPU cloud environments

set -e  # Exit on error

echo "=========================================="
echo "4D Gaussian Splatting Installation Script"
echo "=========================================="
echo ""
echo "Optimized for: runpod/pytorch:2.8.0-py3.11-cuda12.8.1"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.8" | bc) -eq 1 ]]; then
    echo "Error: Python 3.8+ is required"
    exit 1
fi

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    HAS_CUDA=1
else
    echo "Warning: CUDA not detected. Training will be slower."
    HAS_CUDA=0
fi

# Make sure we're not in a conda environment
if [[ ! -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Deactivating conda environment: $CONDA_DEFAULT_ENV"
    conda deactivate
fi

# Check PyTorch is available (should be pre-installed in RunPod)
echo "Checking PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "Warning: PyTorch not found. This script is optimized for RunPod environments."
    echo "Installing PyTorch manually..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
}

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing requirements..."
echo "Installing NumPy 1.24.3 first for compatibility..."
pip install numpy==1.24.3 --force-reinstall

echo "Installing other dependencies..."
pip install -r requirements_runpod.txt --force-reinstall

# Install gsplat for CUDA acceleration
if [ $HAS_CUDA -eq 1 ]; then
    echo "Installing gsplat for CUDA acceleration..."
    pip install gsplat==0.1.11 || echo "Warning: gsplat installation failed. CUDA acceleration will not be available."
fi

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import imageio; print('imageio: OK')"

if [ $HAS_CUDA -eq 1 ]; then
    python3 -c "import gsplat; print('gsplat: OK')" 2>/dev/null || echo "gsplat: Not available (optional)"
fi

# Download sample data (optional)
read -p "Download sample video for testing? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading sample video..."
    mkdir -p data/samples
    wget -O data/samples/sample.mp4 https://www.sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4 2>/dev/null || \
    curl -o data/samples/sample.mp4 https://www.sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4 2>/dev/null || \
    echo "Could not download sample video"
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Quick start:"
echo "  1. Process a video:"
echo "     python tools/preprocess_video.py your_video.mp4 -o dataset/"
echo ""
echo "  2. Train the model:"
echo "     python tools/train.py --data_root dataset/ --out_dir model/ --renderer fast"
echo ""
echo "  3. Render results:"
echo "     python tools/render.py --data_root dataset/ --ckpt model/model_final.pt --out_dir renders/"
echo ""
echo "Note: No environment activation needed - just run commands directly!"
echo ""
