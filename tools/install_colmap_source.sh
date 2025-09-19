#!/usr/bin/env bash
# Build and install COLMAP from source on Ubuntu (with optional CUDA)
# Usage:
#   bash tools/install_colmap_source.sh [--with-cuda]
#
# Notes:
# - Installs dependencies via apt
# - On Ubuntu 22.04 + nvidia-cuda-toolkit, switches to GCC 10 as host compiler
# - Sets QT_QPA_PLATFORM=offscreen for headless runs

set -euo pipefail

WITH_CUDA=0
for arg in "$@"; do
  if [[ "$arg" == "--with-cuda" ]]; then
    WITH_CUDA=1
  fi
done

echo "[COLMAP] Installing build dependencies (requires sudo)..."
sudo apt-get update -y
sudo apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    libcurl4-openssl-dev \
    libmkl-full-dev

if [[ $WITH_CUDA -eq 1 ]]; then
  echo "[COLMAP] Installing CUDA toolkit (Ubuntu package)..."
  sudo apt-get install -y nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc || true
  echo "[COLMAP] Installing GCC-10 for CUDA host compiler workaround (Ubuntu 22.04)..."
  sudo apt-get install -y gcc-10 g++-10 || true
  export CC=/usr/bin/gcc-10
  export CXX=/usr/bin/g++-10
  export CUDAHOSTCXX=/usr/bin/g++-10
fi

echo "[COLMAP] Cloning source..."
rm -rf /tmp/colmap-src
git clone https://github.com/colmap/colmap.git /tmp/colmap-src
cd /tmp/colmap-src
mkdir -p build && cd build

CMAKE_ARGS=( -GNinja -DBLA_VENDOR=Intel10_64lp )
if [[ $WITH_CUDA -eq 1 ]]; then
  CMAKE_ARGS+=( -DCMAKE_CUDA_ARCHITECTURES=native )
fi

echo "[COLMAP] Configuring..."
cmake .. "${CMAKE_ARGS[@]}"
echo "[COLMAP] Building..."
ninja -j"$(nproc)"
echo "[COLMAP] Installing (sudo)..."
sudo ninja install

echo "[COLMAP] Verifying installation..."
QT_QPA_PLATFORM=offscreen colmap -h >/dev/null 2>&1 && echo "[COLMAP] ✓ Installed and runnable (headless)" || echo "[COLMAP] ⚠ Installed but 'colmap -h' failed"

echo "[COLMAP] Done."


