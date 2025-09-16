#!/bin/bash
# RunPod Installation Script for 4D Gaussian Splatting
# This script uses RunPod's base environment (no conda) for maximum compatibility

echo "=============================================="
echo "4D Gaussian Splatting - RunPod Installation"
echo "=============================================="

# Make sure we're not in a conda environment
if [[ ! -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Deactivating conda environment: $CONDA_DEFAULT_ENV"
    conda deactivate
fi

# Check if we're in the right directory
if [[ ! -f "requirements_runpod.txt" ]]; then
    echo "Error: requirements_runpod.txt not found!"
    echo "Make sure you're in the 4DGS_VibeCoded directory"
    echo "Run: cd 4DGS_VibeCoded"
    exit 1
fi

# Check PyTorch version
echo "Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || {
    echo "Error: PyTorch not found in base environment"
    exit 1
}

# Install NumPy first (specific version for compatibility)
echo "Installing NumPy 1.24.3 for compatibility..."
pip install numpy==1.24.3 --force-reinstall

# Install other dependencies
echo "Installing project dependencies..."
pip install -r requirements_runpod.txt --force-reinstall

# Install gsplat for CUDA acceleration
echo "Installing gsplat for CUDA acceleration..."
pip install gsplat==0.1.11 || {
    echo "Warning: gsplat installation failed. CUDA acceleration will not be available."
}

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import imageio; print('imageio: OK')"

# Try to import gsplat
python -c "import gsplat; print('gsplat: OK')" 2>/dev/null || echo "gsplat: Not available (optional)"

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
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
echo "Note: Always use the base environment (no conda activate needed)!"