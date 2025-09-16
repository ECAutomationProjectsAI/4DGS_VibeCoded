#!/bin/bash
# 4D Gaussian Splatting Installation Script
# Complete dependency installation for RunPod environments
# NO virtual environments - uses system Python directly

set -e  # Exit on error

echo "=========================================="
echo "4D Gaussian Splatting Installation"
echo "=========================================="
echo "Environment: RunPod/Cloud GPU"
echo "Using system Python (no venv/conda)"
echo "Target: runpod/pytorch:2.8.0-py3.11-cuda12.8.1"
echo "=========================================="
echo ""

# Use python3/pip3 explicitly for clarity
PYTHON_CMD="python3"
PIP_CMD="pip3"

# Check Python version
echo "Step 1: Checking Python version..."
$PYTHON_CMD --version || {
    echo "Error: Python3 not found!"
    exit 1
}

# Check CUDA
echo ""
echo "Step 2: Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    HAS_CUDA=true
else
    echo "Warning: CUDA not detected. Training will be slower."
    HAS_CUDA=false
fi

# Check if running in conda (and deactivate if so)
if [[ ! -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Warning: Conda environment detected. Deactivating..."
    conda deactivate 2>/dev/null || true
fi

# Check PyTorch
echo ""
echo "Step 3: Checking PyTorch..."
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || {
    echo "PyTorch not found. Installing PyTorch with CUDA support..."
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
}

# Upgrade pip
echo ""
echo "Step 4: Upgrading pip, setuptools, wheel..."
$PIP_CMD install --upgrade pip setuptools wheel

# CRITICAL: Install NumPy first with specific version
echo ""
echo "Step 5: Installing NumPy 1.24.3 (CRITICAL for compatibility)..."
$PIP_CMD uninstall numpy -y 2>/dev/null || true
$PIP_CMD install numpy==1.24.3 --force-reinstall --no-cache-dir

# Install all core dependencies
echo ""
echo "Step 6: Installing all core dependencies..."
echo "  - OpenCV (cv2)"
$PIP_CMD install opencv-python>=4.5.0 --no-cache-dir

echo "  - Image/Video processing"
$PIP_CMD install imageio>=2.9.0 --no-cache-dir
$PIP_CMD install imageio-ffmpeg>=0.4.5 --no-cache-dir
$PIP_CMD install Pillow>=9.0.0 --no-cache-dir

echo "  - Scientific computing"
$PIP_CMD install scipy>=1.7.0 --no-cache-dir
$PIP_CMD install scikit-learn>=0.24.0 --no-cache-dir

echo "  - Utilities"
$PIP_CMD install tqdm>=4.62.0 --no-cache-dir
$PIP_CMD install PyYAML>=5.4.0 --no-cache-dir

echo "  - Data handling"
$PIP_CMD install pandas>=1.3.0 --no-cache-dir
$PIP_CMD install h5py>=3.0.0 --no-cache-dir

echo "  - Visualization"
$PIP_CMD install matplotlib>=3.3.0 --no-cache-dir

# Install gsplat for CUDA acceleration (optional but recommended)
echo ""
echo "Step 7: Installing gsplat for CUDA acceleration..."
if [ "$HAS_CUDA" = true ]; then
    $PIP_CMD install gsplat==0.1.11 --no-cache-dir || {
        echo "Warning: gsplat installation failed."
        echo "This is optional but recommended for faster training."
        echo "The system will fall back to naive renderer."
    }
else
    echo "Skipping gsplat (requires CUDA)"
fi

# Comprehensive verification
echo ""
echo "=========================================="
echo "Step 8: Verifying installation..."
echo "=========================================="

echo ""
echo "Core Dependencies:"
echo "------------------"

# Track failures
FAILED=0

# PyTorch and CUDA
$PYTHON_CMD -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" 2>/dev/null || { echo "✗ PyTorch: FAILED"; FAILED=1; }
$PYTHON_CMD -c "import torch; print(f'✓ CUDA: {torch.cuda.is_available()}')" 2>/dev/null || { echo "✗ CUDA check: FAILED"; FAILED=1; }

# NumPy (critical)
$PYTHON_CMD -c "import numpy; assert numpy.__version__.startswith('1.24'), f'Wrong NumPy version: {numpy.__version__}'; print(f'✓ NumPy: {numpy.__version__}')" 2>/dev/null || { echo "✗ NumPy: FAILED or wrong version"; FAILED=1; }

# OpenCV (critical for video processing)
$PYTHON_CMD -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}')" 2>/dev/null || { echo "✗ OpenCV (cv2): FAILED - Video processing will not work!"; FAILED=1; }

# Image/Video processing
$PYTHON_CMD -c "import imageio; print('✓ imageio: OK')" 2>/dev/null || { echo "✗ imageio: FAILED"; FAILED=1; }
$PYTHON_CMD -c "import imageio_ffmpeg; print('✓ imageio-ffmpeg: OK')" 2>/dev/null || { echo "✗ imageio-ffmpeg: FAILED"; FAILED=1; }
$PYTHON_CMD -c "import PIL; print('✓ Pillow: OK')" 2>/dev/null || { echo "✗ Pillow: FAILED"; FAILED=1; }

# Scientific computing
$PYTHON_CMD -c "import scipy; print('✓ scipy: OK')" 2>/dev/null || { echo "✗ scipy: FAILED"; FAILED=1; }
$PYTHON_CMD -c "import sklearn; print('✓ scikit-learn: OK')" 2>/dev/null || { echo "✗ scikit-learn: FAILED"; FAILED=1; }

# Utilities
$PYTHON_CMD -c "import tqdm; print('✓ tqdm: OK')" 2>/dev/null || { echo "✗ tqdm: FAILED"; FAILED=1; }
$PYTHON_CMD -c "import yaml; print('✓ PyYAML: OK')" 2>/dev/null || { echo "✗ PyYAML: FAILED"; FAILED=1; }

# Data handling
$PYTHON_CMD -c "import pandas; print('✓ pandas: OK')" 2>/dev/null || { echo "✗ pandas: FAILED"; FAILED=1; }
$PYTHON_CMD -c "import h5py; print('✓ h5py: OK')" 2>/dev/null || { echo "✗ h5py: FAILED"; FAILED=1; }

# Visualization
$PYTHON_CMD -c "import matplotlib; print('✓ matplotlib: OK')" 2>/dev/null || { echo "✗ matplotlib: FAILED"; FAILED=1; }

# Optional: gsplat
$PYTHON_CMD -c "import gsplat; print('✓ gsplat: OK (CUDA acceleration available)')" 2>/dev/null || echo "⚠ gsplat: Not installed (optional, will use naive renderer)"

echo ""
if [ $FAILED -eq 0 ]; then
    echo "✓✓✓ All critical dependencies installed successfully! ✓✓✓"
else
    echo "✗✗✗ Some dependencies FAILED! Please check above. ✗✗✗"
    echo "Try running: pip3 install -r requirements_runpod.txt"
fi

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
