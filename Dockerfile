# RunPod base image with PyTorch 2.8.0 and CUDA 12.8
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# CUDA architectures for RunPod GPUs (A100, A6000, RTX 4090, etc.)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1

# System dependencies for Ubuntu 22.04
RUN apt-get update && apt-get install -y \
    git build-essential libgl1 libglib2.0-0 cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /workspace

# Upgrade pip and install build tools
RUN pip3 install --upgrade pip setuptools wheel

# Verify PyTorch installation
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# Install NumPy 1.24.3 (critical for compatibility)
RUN pip3 uninstall numpy -y && \
    pip3 install numpy==1.24.3 --no-cache-dir

# Copy and install requirements
COPY requirements_runpod.txt ./requirements_runpod.txt
RUN pip3 install --no-cache-dir -r requirements_runpod.txt

# Install gsplat for CUDA acceleration (nerfstudio rasterizer)
RUN pip3 install --no-cache-dir gsplat==0.1.11

# Copy project
COPY . /workspace

# Ensure tools and scripts are executable
RUN chmod +x /workspace/tools/*.py || true && \
    chmod +x /workspace/scripts/*.py || true

ENV PYTHONPATH=/workspace

# Default command for RunPod
CMD ["bash", "-c", "echo '4DGS RunPod Container Ready!' && echo '' && echo 'Environment: Ubuntu 22.04 with PyTorch 2.8.0 + CUDA 12.8' && echo '' && echo 'Available commands:' && echo '  - scripts/01_extract_and_map.py: Extract frames and map per-frame-per-camera' && echo '  - scripts/02_calibrate_cameras.py: COLMAP calibration from first mapped frame' && echo '  - scripts/03_train_4dgs.py: Train 4D Gaussian Splatting' && echo '  - tools/render.py: Render trained models' && echo '  - tools/export_ply.py: Export to PLY format' && echo '' && echo 'Example usage:' && echo '  python3 scripts/01_extract_and_map.py /videos -o /data --resize 1280 720' && echo '  python3 scripts/02_calibrate_cameras.py --data_root /data' && echo '  python3 scripts/03_train_4dgs.py --data_root /data --out_dir /model' && echo ''"]
