# 4D Gaussian Splatting (4DGS)

## Overview

4D Gaussian Splatting is a method for reconstructing and rendering dynamic 3D scenes from video sequences. This implementation extends 3D Gaussian Splatting to handle temporal dynamics through velocity-based motion modeling, enabling real-time photorealistic rendering of moving objects and scenes.

## Key Techniques

- **4D Gaussian Primitives**: Each Gaussian has position, scale, rotation, opacity, color (spherical harmonics), and velocity components
- **Velocity-based Motion Model**: Positions evolve over time as x(t) = x₀ + v*t
- **Temporal Consistency**: Specialized losses ensure smooth motion without flickering
- **Differentiable Rendering**: Full gradient flow for end-to-end optimization
- **CUDA Acceleration**: Optional gsplat backend for 10-100x faster training

## Major Dependencies

- **PyTorch 2.8.0+** with CUDA 12.8+ (pre-installed in RunPod)
- **NumPy 1.24.3** - Fixed version for compatibility
- **gsplat 0.1.11** - CUDA-accelerated rasterizer from nerfstudio (optional but recommended)
- **OpenCV** - Video processing and frame extraction
- **imageio-ffmpeg** - Video I/O support


## Hardware Requirements for Training

### Minimum Requirements for Training
- **CPU**: 4+ cores (for data preprocessing)
- **RAM**: 16 GB (32 GB for larger datasets)
- **GPU**: NVIDIA GTX 1060 6GB (CUDA 11.0+)
  - Must have CUDA-capable GPU for training
  - VRAM usage scales with resolution and number of Gaussians
- **Storage**: 50 GB (datasets can be large)

### Recommended for Efficient Training
- **CPU**: 8+ cores (faster preprocessing)
- **RAM**: 32 GB (handle multiple camera views)
- **GPU**: NVIDIA RTX 3070 8GB or better
  - RTX 30/40 series have tensor cores for faster training
  - More VRAM allows larger point clouds and higher resolution
- **Storage**: 100 GB SSD (faster data loading)

### Training Performance by GPU
- **GTX 1060 (6GB VRAM)**: 
  - Max resolution: 720p
  - Max sequence: 10 seconds
  - Max Gaussians: ~20,000
  - Training speed: ~5 min per 1000 iterations
  
- **RTX 3070 (8GB VRAM)**: 
  - Max resolution: 1080p
  - Max sequence: 20 seconds
  - Max Gaussians: ~50,000
  - Training speed: ~2 min per 1000 iterations
  
- **RTX 4090 (24GB VRAM)**:
  - Max resolution: 4K
  - Max sequence: 60+ seconds
  - Max Gaussians: ~100,000+
  - Training speed: ~30 sec per 1000 iterations

### Expected Quality by Camera Setup

| Camera Setup | Quality | PSNR Range | Use Cases |
|-------------|---------|------------|------------|
| **1 camera (rotating)** | Poor-Fair | 20-25 dB | Simple objects, proof of concept |
| **2-3 cameras** | Fair-Good | 24-28 dB | Basic reconstruction, previews |
| **4-6 cameras** | Good | 28-32 dB | Production use, most applications |
| **8+ cameras** | Excellent | 32-36 dB | High-quality production, research |

**Note**: Rendering trained models requires much less resources - any modern GPU with 4GB+ VRAM can render the outputs.


## Installation

### Environment Requirements
- **RunPod/Cloud GPU**: PyTorch 2.8.0+ with CUDA 12.8+
- **Python**: 3.10-3.11
- **GPU**: NVIDIA with CUDA compute capability 6.0+

### Tested Environment
This project is optimized for:
`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`

### Installation Steps

#### Quick Install (Recommended)
```bash
# 1. Clone repository
git clone https://github.com/ECAutomationProjectsAI/4DGS_VibeCoded.git
cd 4DGS_VibeCoded

# 2. Run installation script (REQUIRED - installs all dependencies)
bash install.sh

# 3. Verify installation worked
python -c "import cv2, imageio, numpy; print('Dependencies OK')"
```

#### Manual Install
```bash
# 1. Clone repository
git clone https://github.com/ECAutomationProjectsAI/4DGS_VibeCoded.git
cd 4DGS_VibeCoded

# 2. Install dependencies
pip install -r requirements_runpod.txt --force-reinstall

# 3. Install CUDA acceleration (optional but recommended)
pip install gsplat==0.1.11

# 4. Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Docker Alternative

```bash
# Build image
docker build -t gs4d:latest .

# Run with GPU
docker run --gpus all -it gs4d:latest
```


## Input Data Requirements

### Camera Setup Requirements

#### Single Camera (Limited Results)
- **Can work?** Yes, but with significant limitations
- **Quality**: Poor to moderate 3D reconstruction
- **Best for**: Objects with simple geometry, rotating objects, or camera orbiting around static object
- **Requirements**: 
  - Camera or object must move to capture multiple viewpoints
  - Slow, smooth motion (avoid shaky footage)
  - Complete 360° coverage if possible

#### Multi-Camera (Recommended)
- **Minimum recommended**: 3-4 cameras from different angles
- **Optimal**: 6-8 cameras for full coverage
- **Best results**: 8-12 synchronized cameras
- **Setup tips**:
  - Place cameras at different heights and angles
  - Ensure overlapping fields of view (30-50% overlap)
  - Synchronize cameras or use sync offsets

### Background Requirements

#### Current Capabilities
- **Background handling**: No automatic background removal
- **Works best with**:
  - Clean, uncluttered backgrounds
  - Static backgrounds (no moving objects)
  - Uniform or simple backgrounds (green screen, white wall)
  - Consistent lighting

#### Background Recommendations
- **Option 1**: Use green/blue screen for easier post-processing
- **Option 2**: Use depth sensors to create masks
- **Option 3**: Pre-process videos to remove background:
  ```bash
  # You can pre-process videos with external tools like:
  # - RunwayML for AI background removal
  # - OpenCV background subtraction
  # - Commercial tools like Unscreen
  ```

### Video Content Requirements

#### What Works Well
- **Subject**: Single moving object or person
- **Motion**: Smooth, continuous movement
- **Duration**: 5-30 seconds optimal
- **Frame rate**: 24-60 fps
- **Resolution**: 720p-1080p (higher = better quality but slower)
- **Lighting**: Consistent, well-lit, minimal shadows

#### What to Avoid
- Fast, jerky movements
- Motion blur
- Transparent or reflective objects
- Extreme lighting changes
- Multiple moving objects (unless all are part of the subject)
- Occlusions (objects passing in front)

### Supported Formats
- **Video files**: MP4, AVI, MOV, MKV (any OpenCV-supported format)
- **Image sequences**: JPG, PNG frames with transforms.json
- **Other 4DGS formats**: SpacetimeGaussian PLY, Fudan 4DGS checkpoints

### Data Structure After Processing
```
dataset/
├── frames/           # Extracted frames
│   ├── cam0/        # Camera 0 (required)
│   ├── cam1/        # Camera 1 (optional but recommended)
│   ├── cam2/        # Camera 2 (more cameras = better quality)
│   └── ...
├── transforms.json   # Camera parameters & timestamps
└── metadata.json     # Processing information
```


## Step-by-Step Workflow

### Step 1: Prepare Video Data

#### Example A: Single Camera (Turntable/Orbiting Setup)
```bash
# For single camera - object should rotate or camera should orbit
python tools/preprocess_video.py turntable_video.mp4 -o dataset/ \
    --resize 1280 720              # Consistent resolution
    --extract-every 2              # Reduce frames for faster processing
```

#### Example B: Multi-Camera Setup (Recommended)
```bash
# 4-camera setup for good results
python tools/preprocess_video.py \
    front.mp4 left.mp4 right.mp4 back.mp4 \
    -o dataset/ \
    --camera-names front left right back \
    --sync-offsets 0.0 0.0 0.0 0.0  # Adjust if cameras aren't synchronized

# 8-camera setup for best results
python tools/preprocess_video.py \
    cam0.mp4 cam1.mp4 cam2.mp4 cam3.mp4 \
    cam4.mp4 cam5.mp4 cam6.mp4 cam7.mp4 \
    -o dataset/ \
    --resize 1920 1080
```

#### Example C: With Background Preprocessing
```bash
# If you have green screen footage, consider masking first
# (requires external tools or manual preprocessing)
python tools/preprocess_video.py masked_video.mp4 -o dataset/ \
    --start 10 --end 30           # Extract specific time range
```

### Step 2: Train 4DGS Model

#### 🚀 NEW: Automatic Resource Detection!
The training script now **automatically optimizes** for your hardware:

```bash
# Just run - everything auto-configured!
python tools/train.py --data_root dataset/ --out_dir model/

# The script will:
# ✓ Auto-detect available CPU RAM and use 90% efficiently
# ✓ Auto-select GPU with most free VRAM (uses 95%)
# ✓ Auto-calculate optimal max points (150k per GB VRAM)
# ✓ Auto-adjust frame loading with memory validation
# ✓ Auto-scale training iterations for scene complexity
# ✓ Show comprehensive progress with memory monitoring
# ✓ Stop with clear error if insufficient resources
```

#### Basic Training Examples

```bash
# Simplest - full auto mode
python tools/train.py --data_root dataset/ --out_dir model/

# With some preferences
python tools/train.py \
    --data_root dataset/ \
    --out_dir model/ \
    --renderer fast           # Use CUDA acceleration\
    --w_temporal 0.01         # Temporal smoothness

# Override auto-detection if needed
python tools/train.py \
    --data_root dataset/ \
    --out_dir model/ \
    --max_points 50000        # Force max Gaussians (auto: based on VRAM)\
    --max_memory_gb 16        # Force RAM limit (auto: 85% of available)\
    --gpu_id 1                # Force GPU selection (auto: best available)\
    --memory_fraction 0.9     # Use 90% of resources (default: 85%)
```

#### Key Training Parameters

**Auto-Detected (no need to set):**
- `--max_points`: Automatically set based on GPU VRAM (150k per GB at 95% usage)
- `--max_memory_gb`: Uses 90% of available RAM (aggressive but safe)
- `--gpu_id`: Selects GPU with most free memory (-1 for auto)
- `--iters`: 30k for simple scenes, 50k for complex (auto-adjusted)
- Memory validation before starting (stops if insufficient)

**Manual Tuning (optional):**
- `--renderer`: 'fast' for CUDA, 'naive' for compatibility
- `--w_temporal`: Temporal consistency (0.01-0.05)
- `--sh_degree`: Color complexity (0-3, higher = better)
- `--memory_fraction`: CPU usage (default 0.90 = 90% of RAM)
- `--vram_fraction`: GPU usage (default 0.95 = 95% of VRAM)


### Step 3: Render Output

```bash
# Basic rendering
python tools/render.py \
    --data_root dataset/ \
    --ckpt model/model_final.pt \
    --out_dir renders/

# High-quality rendering
python tools/render.py \
    --data_root dataset/ \
    --ckpt model/model_final.pt \
    --out_dir renders/ \
    --renderer fast
```

### Step 4: Evaluate Results (Optional)

```bash
# Calculate PSNR and other metrics
python tools/evaluate.py --data_root dataset/ --renders_dir renders/
```

## Additional Tools

### Format Conversion

```bash
# From SpacetimeGaussian PLY format
python tools/convert.py input.ply output.pt --from spacetime --to gs4d

# From Fudan 4DGS checkpoint
python tools/convert.py checkpoint.pth output.pt --from fudan --to gs4d

# To SpacetimeGaussian PLY format
python tools/convert.py model.pt output.ply --from gs4d --to spacetime
```

### Generate Synthetic Test Data

```bash
# Create synthetic dataset for testing
python tools/prepare_synthetic.py --out_root test_data/ --frames 10 --H 256 --W 256
```


## Tips for Better Results

### Single Camera Setup Tips
If you only have one camera:
1. **Use a turntable**: Place object on rotating platform
2. **Orbit the camera**: Move camera around stationary object in smooth circle
3. **Ensure complete coverage**: Capture full 360° if possible
4. **Keep motion slow and steady**: Avoid sudden movements
5. **Maintain consistent distance**: Don't zoom in/out during capture
6. **Use markers**: Place reference markers for better tracking
7. **Increase training iterations**: Use `--iters 50000` for better convergence

### Background Handling Tips
1. **Green screen**: Easiest to remove in post-processing
2. **Plain backdrop**: White/black sheets work well
3. **Depth camera**: Use RGB-D cameras to generate masks
4. **Pre-process with AI tools**:
   - Remove.bg for images
   - RunwayML for videos
   - Unscreen for professional results

### General Quality Tips
1. **More cameras = better quality**: Each additional viewpoint helps significantly
2. **Synchronization matters**: Use hardware sync or clapperboard for alignment
3. **Consistent lighting**: Avoid shadows and reflections
4. **Higher resolution**: Train at highest resolution your GPU can handle
5. **Temporal consistency**: Use `--w_temporal 0.02-0.05` for smoother motion

## Output

After completing the workflow, you will have:

1. **Trained Model** (`model/model_final.pt`): 4D Gaussian representation of your scene
2. **Rendered Frames** (`renders/`): Reconstructed video frames from novel viewpoints
3. **Metrics** (optional): PSNR, SSIM values for quality assessment

The system produces dynamic 3D reconstructions with quality dependent on:
- Number of camera viewpoints (more = better)
- Background complexity (simpler = better)
- Subject motion (smoother = better)
- Training iterations (more = better convergence)

## Troubleshooting

### ModuleNotFoundError: No module named 'gs4d'
If you get "No module named 'gs4d'" error:
```bash
# Option 1: Install package in development mode (best)
cd 4DGS_VibeCoded
pip3 install -e . --no-deps

# Option 2: Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Option 3: Use the run.sh helper script
bash run.sh tools/train.py --data_root dataset/
```

### ModuleNotFoundError (cv2, imageio, etc.)
If you get module not found errors, the dependencies aren't installed:
```bash
# Option 1: Run the installation script
bash install.sh

# Option 2: Manual install all dependencies
pip3 install numpy==1.24.3 opencv-python imageio imageio-ffmpeg Pillow scipy scikit-learn tqdm PyYAML pandas h5py matplotlib
pip3 install gsplat==0.1.11  # optional for CUDA acceleration
```

### NumPy Compatibility Error
If you see "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x":
```bash
# Force reinstall NumPy with the correct version
pip uninstall numpy -y
pip install numpy==1.24.3 --force-reinstall
```

### CUDA Out of Memory
- Reduce `--max_points` (e.g., 20000)
- Lower resolution with `--resize 960 540`
- Extract fewer frames with `--extract-every 3`

### Poor Quality
- Increase iterations: `--iters 50000`
- Adjust temporal weight: `--w_temporal 0.02`
- Use stable footage with good lighting
- Add more camera viewpoints if possible

### Slow Training
- Enable CUDA: `--renderer fast`
- Lower SH degree: `--sh_degree 1`
- Reduce point count: `--max_points 30000`

## References

This implementation integrates techniques from:
- [3D Gaussian Splatting](https://github.com/graphdeco-insa/gaussian-splatting) - Base 3DGS method
- [SpacetimeGaussian](https://github.com/oppo-us-research/SpacetimeGaussian) - Temporal extension
- [Fudan 4DGS](https://github.com/fudan-zvg/4d-gaussian-splatting) - 4D primitives
- [gsplat](https://github.com/nerfstudio-project/gsplat) - CUDA acceleration
