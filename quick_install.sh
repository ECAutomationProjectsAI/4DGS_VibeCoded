#!/bin/bash
# Quick installation for RunPod - installs all dependencies directly

echo "Quick Install for 4D Gaussian Splatting"
echo "========================================"

# Install all dependencies in one go
pip3 install --no-cache-dir \
    numpy==1.24.3 \
    opencv-python>=4.5.0 \
    imageio>=2.9.0 \
    imageio-ffmpeg>=0.4.5 \
    Pillow>=9.0.0 \
    scipy>=1.7.0 \
    scikit-learn>=0.24.0 \
    tqdm>=4.62.0 \
    PyYAML>=5.4.0 \
    pandas>=1.3.0 \
    h5py>=3.0.0 \
    matplotlib>=3.3.0

# Install gsplat (optional)
pip3 install gsplat==0.1.11 --no-cache-dir || echo "gsplat not installed (optional)"

# Quick test
echo ""
echo "Testing installation..."
python3 -c "import cv2, imageio, numpy; print('✓ All core dependencies installed!')"

echo ""
echo "Ready! You can now run:"
echo "  python3 tools/preprocess_video.py input/ -o dataset/"