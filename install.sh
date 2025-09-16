#!/bin/bash
# 4D Gaussian Splatting Installation Script
# Supports Linux, WSL2, and RunPod environments

set -e  # Exit on error

echo "=========================================="
echo "4D Gaussian Splatting Installation Script"
echo "=========================================="

# Detect environment
if [ -f /run/secrets/kubernetes.io/serviceaccount/token ]; then
    echo "Detected: RunPod environment"
    ENV_TYPE="runpod"
elif grep -q Microsoft /proc/version 2>/dev/null; then
    echo "Detected: WSL2 environment"
    ENV_TYPE="wsl2"
elif [ "$(uname)" == "Linux" ]; then
    echo "Detected: Linux environment"
    ENV_TYPE="linux"
else
    echo "Detected: Unknown environment"
    ENV_TYPE="unknown"
fi

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

# Check if using conda
if command -v conda &> /dev/null; then
    echo "Conda detected. Creating conda environment..."
    # Create conda environment if it doesn't exist
    if ! conda env list | grep -q "gs4d"; then
        conda create -n gs4d python=3.10 -y
    fi
    # Activate conda environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate gs4d
    USING_CONDA=1
else
    echo "Conda not detected. Using venv..."
    # Create virtual environment
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    # Activate virtual environment
    source venv/bin/activate
    USING_CONDA=0
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA if available
echo "Installing PyTorch..."
if [ $USING_CONDA -eq 1 ]; then
    # Use conda for PyTorch installation
    if [ $HAS_CUDA -eq 1 ]; then
        conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
    else
        conda install pytorch==2.2.2 torchvision==0.17.2 cpuonly -c pytorch -y
    fi
else
    # Use pip for PyTorch installation
    if [ $HAS_CUDA -eq 1 ]; then
        pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt || {
    echo "Installing requirements individually..."
    pip install 'numpy<2.0'
    pip install 'imageio>=2.9.0'
    pip install 'imageio-ffmpeg>=0.4.5'
    pip install 'opencv-python>=4.5.0'
    pip install 'Pillow>=9.0.0'
    pip install 'scipy>=1.7.0'
    pip install 'scikit-learn>=0.24.0'
    pip install 'tqdm>=4.62.0'
    pip install 'PyYAML>=5.4.0'
    pip install 'pandas>=1.3.0'
    pip install 'h5py>=3.0.0'
    pip install 'matplotlib>=3.3.0'
}

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
echo "To activate the environment:"
if [ $USING_CONDA -eq 1 ]; then
    echo "  conda activate gs4d"
else
    echo "  source venv/bin/activate"
fi
echo ""
echo "Quick start:"
echo "  1. Process a video:"
echo "     python tools/preprocess_video.py your_video.mp4 -o dataset/"
echo ""
echo "  2. Train the model:"
echo "     python tools/train.py --data_root dataset/ --out_dir model/"
echo ""
echo "  3. Render results:"
echo "     python tools/render.py --checkpoint model/model_final.pt --out_dir renders/"
echo ""
echo "For Docker usage, run:"
echo "  docker build -t gs4d:latest ."
echo ""