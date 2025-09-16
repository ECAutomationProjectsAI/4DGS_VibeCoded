# GPU-enabled base image with CUDA toolchain for building extensions
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Pre-set CUDA arch list to build CUDA extensions
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git build-essential libgl1 libglib2.0-0 cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /workspace

# Install Python build tools
RUN pip3 install --upgrade pip setuptools wheel cmake ninja

# Install PyTorch with CUDA support
RUN pip3 install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# Copy and install requirements
COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt || \
    pip3 install --no-cache-dir \
    'numpy<2.0' \
    'imageio>=2.9.0' \
    'imageio-ffmpeg>=0.4.5' \
    'opencv-python>=4.5.0' \
    'Pillow>=9.0.0' \
    'scipy>=1.7.0' \
    'scikit-learn>=0.24.0' \
    'tqdm>=4.62.0' \
    'PyYAML>=5.4.0' \
    'matplotlib>=3.3.0'

# Install gsplat for CUDA acceleration
# This is the high-performance rasterizer from nerfstudio
RUN pip3 install --no-cache-dir gsplat==0.1.11

# Copy project
COPY . /workspace

# Ensure tools are executable
RUN chmod +x /workspace/tools/*.py || true

ENV PYTHONPATH=/workspace

# Default command shows available tools
CMD ["bash", "-c", "echo '4D Gaussian Splatting Container Ready!' && echo '' && echo 'Available tools:' && echo '  - preprocess_video.py: Process videos into datasets' && echo '  - train.py: Train 4DGS models' && echo '  - render.py: Render trained models' && echo '  - convert.py: Convert between formats' && echo '' && echo 'Example usage:' && echo '  python3 tools/preprocess_video.py video.mp4 -o dataset/' && echo '  python3 tools/train.py --data_root dataset/ --out_dir model/' && echo ''"]
