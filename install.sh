#!/bin/bash
# 4D Gaussian Splatting Installation Script
# Optimized for: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
# NO virtual environments - uses system Python directly

set -e  # Exit on error

echo "=========================================="
echo "4D Gaussian Splatting Installation"
echo "=========================================="
echo "Environment: RunPod Ubuntu 22.04"
echo "Target: runpod/pytorch:2.8.0-py3.11-cuda12.8.1"
echo "Using system Python (no venv/conda)"
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

# RunPod environment check
if [[ ! -f /etc/os-release ]] || ! grep -q "Ubuntu 22.04" /etc/os-release 2>/dev/null; then
    echo "Warning: This script is optimized for Ubuntu 22.04 (RunPod environment)"
    echo "Current system: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2 2>/dev/null || echo 'Unknown')"
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
echo "  Note: PyTorch 2.8+ comes with NumPy 2.x which is incompatible."
echo "  We need to downgrade to NumPy 1.24.3 for compatibility."

# First check current numpy version
$PYTHON_CMD -c "import numpy; print(f'  Current NumPy: {numpy.__version__}')" 2>/dev/null || echo "  NumPy not installed"

# Complete NumPy cleanup and reinstall for Ubuntu
echo "  Performing complete NumPy cleanup..."
# Uninstall all numpy versions
$PIP_CMD uninstall numpy -y 2>/dev/null || true
$PIP_CMD uninstall numpy -y 2>/dev/null || true  # Run twice to be sure

# Clear pip cache (Ubuntu location)
rm -rf ~/.cache/pip/* 2>/dev/null || true
$PIP_CMD cache purge 2>/dev/null || true

# Install specific numpy version for RunPod/Ubuntu
echo "  Installing NumPy 1.24.3..."
$PIP_CMD install numpy==1.24.3 --force-reinstall --no-deps --no-cache-dir || {
    echo "  Trying alternative NumPy installation method..."
    $PIP_CMD install 'numpy>=1.24,<1.25' --force-reinstall --no-cache-dir
}

# Verify it worked
$PYTHON_CMD -c "import numpy; assert numpy.__version__.startswith('1.24'), f'NumPy {numpy.__version__} installed, need 1.24.x'; print(f'  ✓ NumPy {numpy.__version__} installed successfully')" || {
    echo "  ✗ Failed to install NumPy 1.24.3!"
    echo "  Running fix script as fallback..."
    $PYTHON_CMD fix_numpy.py || echo "  Fix script failed. Manual intervention required."
}

# Install all core dependencies via requirements (respects pinned NumPy)
echo ""
echo "Step 6: Installing all core dependencies (requirements_runpod.txt)..."
$PIP_CMD install --no-cache-dir -r requirements_runpod.txt || true

# Re-pin NumPy in case resolver upgraded it
$PIP_CMD install numpy==1.24.3 --force-reinstall --no-deps --no-cache-dir || true

# Cleanup stray files that may have been created by shell redirection in older scripts
rm -f =* 2>/dev/null || true

# Try to install COLMAP via apt if not already available
echo ""
echo "Step 6.5: Ensuring COLMAP is installed (for multi-view calibration)..."
if ! command -v colmap &> /dev/null; then
    echo "COLMAP not found in PATH. Attempting installation via apt..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y && \
    apt-get install -y colmap || {
        echo "⚠ Could not install COLMAP via apt (may not be available on this image)."
        echo "  You can manually install from: https://github.com/colmap/colmap/releases"
    }
else
    echo "✓ COLMAP already installed"
fi

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

# Install the gs4d package in development mode
echo ""
echo "Step 8: Installing gs4d package in development mode..."
if [ -f "setup.py" ]; then
    $PIP_CMD install -e . --no-deps
    echo "✓ gs4d package installed in development mode"
else
    echo "⚠ Warning: setup.py not found. Running from source directory."
    echo "  You may need to run scripts with: python3 tools/script.py"
    echo "  Or add the project directory to PYTHONPATH:"
    echo "  export PYTHONPATH=\$PYTHONPATH:\$(pwd)"
fi

# Comprehensive verification
echo ""
echo "=========================================="
echo "Step 9: Verifying installation..."
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

# Verify COLMAP presence (optional)
command -v colmap &> /dev/null && echo "✓ COLMAP: $(colmap -h | head -n 1 2>/dev/null || echo 'found')" || echo "⚠ COLMAP: Not found (multi-view calibration will be skipped unless --skip_colmap is used)"

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
$PYTHON_CMD -c "import psutil; print('✓ psutil: OK')" 2>/dev/null || { echo "✗ psutil: FAILED"; FAILED=1; }

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
echo "Option A: If gs4d package was installed:"
echo "  1. Process: python3 tools/preprocess_video.py your_video.mp4 -o dataset/"
echo "  2. Train:   python3 tools/train.py --data_root dataset/ --out_dir model/"
echo "  3. Render:  python3 tools/render.py --data_root dataset/ --ckpt model/model_final.pt --out_dir renders/"
echo ""
echo "Option B: If running from source (without package install):"
echo "  Set PYTHONPATH first: export PYTHONPATH=\$PYTHONPATH:\$(pwd)"
echo "  Or use run.sh:        bash run.sh tools/train.py --data_root dataset/"
echo ""
echo "Note: No environment activation needed - just run commands directly!"
echo ""
