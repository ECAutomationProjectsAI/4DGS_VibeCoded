# RunPod-compatible base image with PyTorch 2.8.0 and CUDA 12.8
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Pre-set CUDA arch list to build CUDA extensions
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git build-essential libgl1 libglib2.0-0 cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /workspace

# Install Python build tools
RUN pip install --upgrade pip setuptools wheel cmake ninja

# PyTorch is pre-installed in the base image, no need to install
RUN python -c "import torch; print(f'PyTorch {torch.__version__} is ready')"

# Install NumPy first for compatibility
RUN pip install numpy==1.24.3 --force-reinstall

# Copy and install requirements
COPY requirements_runpod.txt ./requirements_runpod.txt
RUN pip install --no-cache-dir -r requirements_runpod.txt

# Install gsplat for CUDA acceleration
# This is the high-performance rasterizer from nerfstudio
RUN pip install --no-cache-dir gsplat==0.1.11

# Copy project
COPY . /workspace

# Ensure tools are executable
RUN chmod +x /workspace/tools/*.py || true

ENV PYTHONPATH=/workspace

# Default command shows available tools
CMD ["bash", "-c", "echo '4D Gaussian Splatting Container Ready!' && echo '' && echo 'Available tools:' && echo '  - preprocess_video.py: Process videos into datasets' && echo '  - train.py: Train 4DGS models' && echo '  - render.py: Render trained models' && echo '  - convert.py: Convert between formats' && echo '' && echo 'Example usage:' && echo '  python3 tools/preprocess_video.py video.mp4 -o dataset/' && echo '  python3 tools/train.py --data_root dataset/ --out_dir model/' && echo ''"]
