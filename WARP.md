# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a production-ready implementation of 4D Gaussian Splatting (4DGS) with CUDA acceleration, temporal consistency, and comprehensive video processing capabilities. The system transforms videos into dynamic 3D representations using differentiable Gaussian splatting techniques with temporal modeling.

## Common Development Commands

### Environment Setup (Ubuntu/RunPod)
```bash
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install gsplat==0.1.11  # CUDA acceleration
```

### Core Workflow Commands

#### 1. Process Videos (Folder) to Dataset
```bash
# Folder of videos (auto camera names from filenames)
python tools/preprocess_video.py videos/ -o dataset/ --resize 1280 720 --extract-every 1

# With time range and subsampling
python tools/preprocess_video.py videos/ -o dataset/ --start_frame 0 --end_frame 2000 --extract-every 2
```

#### 2. Train 4DGS Model
```bash
# Basic training
python tools/train.py --data_root dataset/ --out_dir model/

# Production training with CUDA acceleration
python tools/train.py --data_root dataset/ --out_dir model/ --renderer fast --iters 30000 --w_temporal 0.01

# Limited GPU memory (< 8GB)
python tools/train.py --data_root dataset/ --out_dir model/ --max_points 20000 --sh_degree 1
```

#### 3. Render Results
```bash
# Basic rendering
python tools/render.py --checkpoint model/model_final.pt --out_dir renders/

# High-quality rendering
python tools/render.py --checkpoint model/model_final.pt --out_dir renders/ --resolution 1920 1080 --fps 60
```

#### 4. Format Conversion
```bash
# From SpacetimeGaussian PLY
python tools/convert.py input.ply output.pt --from spacetime --to gs4d

# To SpacetimeGaussian PLY
python tools/convert.py model.pt output.ply --from gs4d --to spacetime
```

### Testing & Validation
```bash
# Generate synthetic test data
python tools/prepare_synthetic.py --out_root test_data/

# Quick test training (100 iterations)
python tools/train.py --data_root test_data/ --out_dir test_model/ --iters 100

# Verify CUDA and dependencies
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import gsplat; print('gsplat installed')"
```

### Docker Commands
```bash
# Build image
docker build -t gs4d:latest .

# Run with GPU support
```bash
docker run --gpus all -v $(pwd)/videos:/videos -v $(pwd)/output:/output gs4d:latest \
  python3 /workspace/tools/preprocess_video.py /videos -o /output/dataset/

# Docker Compose
docker-compose up
```
## Code Architecture

### Core Library Structure (`gs4d/`)

#### `gaussians.py` - 4D Gaussian Model
- **Key Class**: `GaussianModel4D` - Main model with velocity-based dynamics
- **Motion Model**: x(t) = x₀ + v*t (position evolves linearly with velocity)
- **Key Methods**:
  - `position_at_time(t)`: Compute positions at timestamp t
  - `densify_clone()`: Clone Gaussians for densification
  - `prune()`: Remove low-opacity Gaussians
- **Parameters**: xyz (positions), velocity, scales, quaternions, opacity, SH coefficients

#### `renderer.py` & `fast_renderer.py` - Rendering Engines
- **Naive Renderer**: Pure PyTorch differentiable implementation (slower but always works)
- **Fast Renderer**: CUDA-accelerated using gsplat backend (10-100x faster)
- **Automatic Fallback**: System tries fast renderer first, falls back to naive if unavailable
- **Key Function**: `forward_splat()` - Main rendering pipeline

#### `video_processor.py` - Video Pipeline
- **Classes**: `VideoProcessor`, `CameraConfig`, `VideoDataset`
- **Features**:
  - Multi-format support (MP4, AVI, MOV, MKV)
  - Multi-camera synchronization
  - Automatic calibration extraction
  - COLMAP export capability
- **Output**: Generates `transforms.json` for training

#### `temporal_losses.py` - Temporal Consistency
- **Key Class**: `TemporalConsistencyLoss`
- **Components**:
  - Velocity smoothness regularization
  - Position smoothness across frames
  - Appearance consistency
- **Purpose**: Ensures smooth motion without flickering

#### `converters/` - Format Conversion
- **Modules**: `spacetime.py`, `fudan.py`
- **Supports**: SpacetimeGaussian PLY, Fudan 4DGS checkpoints
- **Bidirectional**: Can convert to/from multiple formats

### Training Pipeline (`tools/train.py`)

The training script implements a sophisticated pipeline:

1. **Initialization**: Random point cloud from first frame
2. **Optimization Loop**:
   - Sample random frame
   - Compute time-dependent positions
   - Render with selected backend (naive/fast)
   - Compute losses (L1, SSIM, temporal)
   - Backpropagate
3. **Adaptive Control**:
   - Densification (clone/split based on gradients)
   - Pruning (remove low-opacity points)
   - SH degree growth
4. **Checkpointing**: Save models at intervals

### Key Parameters

#### Training Parameters
- `--iters`: Training iterations (default: 2000, production: 30000+)
- `--lr`: Learning rate (default: 0.01)
- `--renderer`: 'naive' or 'fast' (CUDA acceleration)
- `--w_temporal`: Temporal consistency weight (0.01-0.05)
- `--w_ssim`: SSIM loss weight (default: 0.2)
- `--sh_degree`: Spherical harmonics degree (0-3)
- `--max_points`: Maximum Gaussians (20000-100000)

#### Densification Parameters
- `--densify_grad_thresh`: Gradient threshold for cloning (default: 0.001)
- `--prune_opacity_thresh`: Opacity threshold for pruning (default: 0.01)
- `--densification_interval`: How often to densify (default: 100)

## Performance Optimization

### GPU Memory Management
```bash
# For < 8GB VRAM
--max_points 20000 --sh_degree 1 --resize 960 540

# For 8-16GB VRAM
--max_points 50000 --sh_degree 2 --resize 1280 720

# For 24GB+ VRAM
--max_points 100000 --sh_degree 3 --resize 1920 1080
```

### Speed Optimization
- Always use `--renderer fast` when gsplat is available
- Extract fewer frames: `--extract-every 3`
- Lower SH degree for faster training: `--sh_degree 1`

### Quality Optimization
- Increase iterations: `--iters 50000`
- Higher temporal weight: `--w_temporal 0.02`
- More densification: `--densify_until_iter 30000`
- Use multiple camera views when available

## File Formats

### Input Data Structure
```
dataset/
├── frames/
│   ├── cam0/
│   │   ├── 00000.png
│   │   └── ...
│   └── cam1/ (if multi-camera)
├── transforms.json    # Camera parameters & timestamps
└── metadata.json      # Processing metadata
```

### transforms.json Format
```json
{
  "frames": [
    {
      "file_path": "frames/cam0/00000.png",
      "transform_matrix": [[4x4 matrix]],
      "time": 0.0,
      "camera_angle_x": 1.0
    }
  ]
}
```

### Model Checkpoint Format
- PyTorch state dict with Gaussian parameters
- Contains: xyz, velocity, scales, quaternions, opacity, SH coefficients
- Saved as `.pt` files

## Troubleshooting Guide

### CUDA Issues
- **Out of Memory**: Reduce `--max_points`, lower resolution, extract fewer frames
- **CUDA Not Available**: Check PyTorch installation, verify with `nvidia-smi`
- **gsplat Not Found**: Install with `pip install gsplat==0.1.11`, system will fallback to naive renderer

### Video Processing Issues
- **ffmpeg Error**: Install with `apt-get install ffmpeg` (Linux) or through conda
- **Unsupported Format**: Convert to MP4 first
- **Memory Error**: Process shorter segments or reduce resolution

### Training Issues
- **Poor Quality**: Increase iterations, adjust temporal weight, ensure stable footage
- **Slow Training**: Enable CUDA renderer, reduce point count, lower SH degree
- **Divergence**: Lower learning rate, check data normalization

## Development Tips

### Adding New Features
- Renderers should inherit from base classes in `renderer.py`
- New losses go in `losses.py` or `temporal_losses.py`
- Format converters go in `gs4d/converters/`
- Command-line tools go in `tools/`

### Testing Changes
1. Always test with synthetic data first: `python tools/prepare_synthetic.py`
2. Verify CUDA compatibility if modifying renderers
3. Check memory usage with different GPU configurations
4. Test format conversions bidirectionally

### Performance Profiling
```python
# Add to training loop
import torch.profiler
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    # training step
prof.export_chrome_trace("trace.json")
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: Select GPU (e.g., "0" for first GPU)
- `TORCH_CUDA_ARCH_LIST`: CUDA architectures for compilation
- `FORCE_CUDA`: Force CUDA compilation even without GPU

## Dependencies

### Core Requirements
- Python 3.8-3.10
- PyTorch 2.2.2 with CUDA 12.1
- CUDA Toolkit 11.8+
- gsplat 0.1.11 (optional but recommended)

### Key Python Packages
- `imageio`, `imageio-ffmpeg`: Video I/O
- `opencv-python`: Image processing
- `scipy`, `scikit-learn`: Scientific computing
- `tqdm`: Progress bars
- `numpy < 2.0`: Core numerics (version constraint important)

## Project Status

This is a production-ready implementation with:
- ✅ Complete video processing pipeline
- ✅ CUDA acceleration via gsplat
- ✅ Temporal consistency modeling
- ✅ Multi-format support
- ✅ Docker containerization
- ✅ Comprehensive documentation

Ready for deployment on RunPod, local machines, or cloud GPU instances.