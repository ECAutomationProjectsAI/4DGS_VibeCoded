#!/bin/bash
# 4D Gaussian Splatting Installation Script for RunPod
# NO virtual environments - uses system Python directly

set -e  # Exit on error

echo "=========================================="
echo "4D Gaussian Splatting Installation"
echo "=========================================="
echo "Environment: RunPod/Cloud GPU"
echo "Using system Python (no venv/conda)"
echo ""

# Use python3 explicitly
PYTHON_CMD="python3"
PIP_CMD="pip3"

# Check Python version
echo "Checking Python version..."
$PYTHON_CMD --version

# Check CUDA
echo "\nChecking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "Warning: CUDA not detected. Training will be slower."
fi

# Check if running in conda (and deactivate if so)
if [[ ! -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Warning: Conda environment detected. Deactivating..."
    conda deactivate 2>/dev/null || true
fi

# Check PyTorch
echo "\nChecking PyTorch..."
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "PyTorch not found. Installing..."
    $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cu118
}

# Upgrade pip
echo "\nUpgrading pip..."
$PIP_CMD install --upgrade pip setuptools wheel

# Install NumPy first (specific version for compatibility)
echo "\nInstalling NumPy 1.24.3 for compatibility..."
$PIP_CMD uninstall numpy -y 2>/dev/null || true
$PIP_CMD install numpy==1.24.3 --force-reinstall --no-cache-dir

# Install OpenCV
echo "\nInstalling OpenCV..."
$PIP_CMD install opencv-python>=4.5.0 --no-cache-dir

# Install all other dependencies
echo "\nInstalling all dependencies..."
$PIP_CMD install \
    imageio>=2.9.0 \
    imageio-ffmpeg>=0.4.5 \
    Pillow>=9.0.0 \
    scipy>=1.7.0 \
    scikit-learn>=0.24.0 \
    tqdm>=4.62.0 \
    PyYAML>=5.4.0 \
    pandas>=1.3.0 \
    h5py>=3.0.0 \
    matplotlib>=3.3.0 \
    --no-cache-dir

# Install gsplat for CUDA acceleration
echo "\nInstalling gsplat for CUDA acceleration..."
$PIP_CMD install gsplat==0.1.11 --no-cache-dir || echo "Warning: gsplat installation failed (optional)"

# Verify installation
echo "\n=========================================="
echo "Verifying installation..."
echo "=========================================="

echo "\nChecking core dependencies:"
$PYTHON_CMD -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || echo "✗ PyTorch FAILED"
$PYTHON_CMD -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" || echo "✗ CUDA check FAILED"
$PYTHON_CMD -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')" || echo "✗ NumPy FAILED"
$PYTHON_CMD -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}')" || echo "✗ OpenCV FAILED"
$PYTHON_CMD -c "import imageio; print('✓ imageio: OK')" || echo "✗ imageio FAILED"
$PYTHON_CMD -c "import PIL; print('✓ Pillow: OK')" || echo "✗ Pillow FAILED"
$PYTHON_CMD -c "import scipy; print('✓ scipy: OK')" || echo "✗ scipy FAILED"
$PYTHON_CMD -c "import sklearn; print('✓ scikit-learn: OK')" || echo "✗ scikit-learn FAILED"
$PYTHON_CMD -c "import tqdm; print('✓ tqdm: OK')" || echo "✗ tqdm FAILED"
$PYTHON_CMD -c "import yaml; print('✓ PyYAML: OK')" || echo "✗ PyYAML FAILED"
$PYTHON_CMD -c "import pandas; print('✓ pandas: OK')" || echo "✗ pandas FAILED"
$PYTHON_CMD -c "import h5py; print('✓ h5py: OK')" || echo "✗ h5py FAILED"
$PYTHON_CMD -c "import matplotlib; print('✓ matplotlib: OK')" || echo "✗ matplotlib FAILED"
$PYTHON_CMD -c "import gsplat; print('✓ gsplat: OK')" 2>/dev/null || echo "⚠ gsplat: Not available (optional)"

echo "\n=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Quick start commands:"
echo ""
echo "1. Process a video:"
echo "   python3 tools/preprocess_video.py your_video.mp4 -o dataset/"
echo ""
echo "2. Train the model:"
echo "   python3 tools/train.py --data_root dataset/ --out_dir model/ --renderer fast"
echo ""
echo "3. Render results:"
echo "   python3 tools/render.py --data_root dataset/ --ckpt model/model_final.pt --out_dir renders/"
echo ""
echo "Note: Use python3 directly - no environment activation needed!"
echo ""
