# 4D Gaussian Splatting Pipeline Guide

## Overview

This guide documents the complete pipeline for 4D Gaussian Splatting (4DGS), from multi-view video capture to final playable output. The implementation combines techniques from multiple state-of-the-art papers:

- **SpacetimeGaussian**: Temporal opacity and polynomial motion trajectories
- **4K4D**: Real-time rendering with depth peeling and image blending
- **FreeTimeGS**: Explicit time assignments and velocity annealing
- **Diffuman4D**: Multi-view video diffusion for view synthesis
- **EasyVolcap**: Unified framework for volumetric capture

## Pipeline Architecture

```
Multi-View Videos → Frame Extraction → Camera Calibration → 4DGS Training → Export
      ↓                    ↓                   ↓                ↓            ↓
 [.mp4/.avi]          [Frames]         [COLMAP/SfM]      [Checkpoints]  [PLY/4DGS]
```

## Complete Workflow

### 1. Data Capture Requirements

**Multi-View Setup:**
- **Minimum**: 4 cameras for basic 360° coverage
- **Recommended**: 8-12 cameras for high quality
- **Optimal**: 18-24 cameras (as used in Neural3DV dataset)

**Video Specifications:**
- Resolution: 1080p minimum, 4K preferred
- Frame rate: 30 FPS minimum, 60 FPS for fast motion
- Format: MP4, AVI, MOV supported
- Synchronization: Hardware sync preferred, software sync possible

### 2. Preprocessing Pipeline

#### Step 2.1: Multi-View Video Processing

```bash
python tools/preprocess_multiview.py \
    --videos video1.mp4 video2.mp4 video3.mp4 video4.mp4 \
    --output processed_data \
    --camera_names cam0 cam1 cam2 cam3 \
    --fps 30 \
    --skip_frames 1 \
    --use_gpu
```

**What happens:**
1. **Frame Extraction**: Videos → synchronized frames
2. **Camera Calibration**: COLMAP SfM for intrinsics/extrinsics
3. **Time Synchronization**: Align timestamps across cameras
4. **Transforms Generation**: Create `transforms.json` for training

#### Step 2.2: Optional Enhancements

**Mask Generation** (for foreground/background separation):
```bash
python tools/generate_masks.py \
    --input processed_data/frames \
    --output processed_data/masks \
    --method sam  # or rembg, u2net
```

**Depth Estimation** (for better initialization):
```bash
python tools/estimate_depth.py \
    --input processed_data \
    --method midas  # or dpt, zoe
```

### 3. Training 4D Gaussian Splatting

#### Basic Training
```bash
python tools/train.py \
    --data_root processed_data \
    --out_dir outputs/my_4dgs \
    --iters 30000 \
    --sh_degree 3 \
    --device cuda
```

#### Advanced Training with Temporal Consistency
```bash
python tools/train.py \
    --data_root processed_data \
    --out_dir outputs/my_4dgs_advanced \
    --iters 50000 \
    --sh_degree 3 \
    --w_temporal 0.01 \
    --temporal_window 3 \
    --densify_grad_thresh 1e-3 \
    --renderer fast \
    --device cuda
```

**Key Parameters:**
- `--iters`: Training iterations (30k minimum, 50k+ for quality)
- `--sh_degree`: Spherical harmonics degree (3 recommended)
- `--w_temporal`: Temporal consistency weight
- `--densify_grad_thresh`: Threshold for Gaussian densification
- `--renderer`: Choose `fast` (GPU) or `naive` (CPU-friendly)

### 4. Rendering and Visualization

#### Render Video Sequence
```bash
python tools/render.py \
    --data_root processed_data \
    --ckpt outputs/my_4dgs/ckpt_30000.pt \
    --out_dir renders/my_sequence \
    --renderer fast
```

### 5. Export Formats

#### Export to PLY (Single Frame)
```bash
python tools/export_ply.py \
    --ckpt outputs/my_4dgs/ckpt_30000.pt \
    --output exports/frame_0.ply \
    --format ply \
    --time 0.0
```

#### Export PLY Sequence (Animation)
```bash
python tools/export_ply.py \
    --ckpt outputs/my_4dgs/ckpt_30000.pt \
    --output exports/sequence \
    --format ply_sequence \
    --num_frames 100 \
    --time_min -0.5 \
    --time_max 0.5
```

#### Export 4DGS Format (Complete Temporal Data)
```bash
python tools/export_ply.py \
    --ckpt outputs/my_4dgs/ckpt_30000.pt \
    --output exports/my_model.4dgs \
    --format 4dgs
```

## Technical Details

### Data Flow Architecture

```python
# 1. Video Input
videos = [cam0.mp4, cam1.mp4, ...]
    ↓
# 2. Frame Extraction + Sync
frames = extract_frames(videos, fps=30)
times = synchronize_timestamps(frames)
    ↓
# 3. Camera Calibration (COLMAP)
cameras = {
    'intrinsics': K,  # [fx, fy, cx, cy]
    'extrinsics': [R, t],  # Rotation, translation
}
    ↓
# 4. 4D Gaussian Parameters
gaussians = {
    'position': xyz + velocity * t,  # Motion model
    'rotation': quaternion,
    'scale': [sx, sy, sz, st],  # Spatial + temporal
    'opacity': α(t),  # Time-varying
    'color': SH coefficients,
}
    ↓
# 5. Rendering
image = splat(gaussians, camera, time)
```

### Key Innovations from Papers

**From SpacetimeGaussian:**
- Temporal opacity: `α(t) = αs · exp(-st · |t-μt|²)`
- Polynomial motion: `μ(t) = Σ bk(t-μt)^k`
- Feature splatting with MLP decoder

**From 4K4D:**
- Depth peeling for correct ordering
- Image blending from nearest views
- FP16 streaming for speed

**From FreeTimeGS:**
- Explicit time assignment per Gaussian
- Velocity annealing during training
- 4D regularization with stop-gradient

**From Diffuman4D:**
- Multi-view video diffusion for novel views
- Sliding window denoising
- Human skeleton conditioning

## Output Formats

### PLY Files
Standard 3D Gaussian Splatting format with extensions:
- **Position** (x, y, z)
- **Velocity** (vx, vy, vz) as normals
- **Time** (t, scale_t) for temporal extent
- **Appearance** (SH coefficients)
- **Opacity** (α)

### 4DGS Format
Custom format preserving full temporal information:
- Complete parameter tensors
- Velocity fields
- Temporal scales
- Ready for direct playback

### Video Export
Rendered frames as image sequence or video file:
- Supports custom camera paths
- Time interpolation
- Multi-view synthesis

## Performance Optimization

### Memory Management
- **Auto-detection**: Automatically configures based on available RAM/VRAM
- **Streaming**: Load frames in batches for large datasets
- **Compression**: Optional frame compression for storage

### GPU Acceleration
- **CUDA Renderer**: Fast differentiable splatting
- **Multi-GPU**: Distributed training support
- **Mixed Precision**: FP16 for memory efficiency

### Training Speed
- **Typical Times**:
  - Small scene (100 frames): ~30 min on RTX 4090
  - Medium scene (300 frames): ~1 hour
  - Large scene (1000+ frames): 2-4 hours

## Troubleshooting

### Common Issues

**Out of Memory:**
```bash
# Reduce max points
--max_points 100000

# Reduce frames
--max_frames 100

# Lower resolution in preprocessing
```

**COLMAP Fails:**
```bash
# Use simpler camera model
--camera_model PINHOLE

# Provide manual calibration
--calibration calib.json
```

**Poor Quality:**
```bash
# Increase iterations
--iters 50000

# Adjust densification
--densify_grad_thresh 5e-4

# Enable temporal consistency
--w_temporal 0.05
```

## Best Practices

1. **Data Capture**:
   - Ensure good lighting and minimal motion blur
   - Use hardware sync for cameras if possible
   - Capture calibration board for accurate intrinsics

2. **Preprocessing**:
   - Start with lower resolution for testing
   - Use every 2nd or 3rd frame initially
   - Verify COLMAP reconstruction visually

3. **Training**:
   - Monitor loss curves for convergence
   - Save checkpoints frequently
   - Start with default parameters, then tune

4. **Export**:
   - Use PLY for compatibility with viewers
   - Use 4DGS format for full temporal playback
   - Generate previews at multiple time points

## Example End-to-End Pipeline

```bash
# 1. Preprocess videos
python tools/preprocess_multiview.py \
    --videos data/cam*.mp4 \
    --output processed \
    --use_gpu

# 2. Train model
python tools/train.py \
    --data_root processed \
    --out_dir model \
    --iters 30000

# 3. Export for viewing
python tools/export_ply.py \
    --ckpt model/ckpt_final.pt \
    --output exports/sequence \
    --format ply_sequence \
    --num_frames 100

# 4. Render video
python tools/render.py \
    --data_root processed \
    --ckpt model/ckpt_final.pt \
    --out_dir renders
```

## References

- [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [SpacetimeGaussian](https://github.com/oppo-us-research/SpacetimeGaussians)
- [4K4D](https://zju3dv.github.io/4k4d/)
- [EasyVolcap](https://github.com/zju3dv/EasyVolcap)

## Support

For issues or questions, please check:
1. This pipeline guide
2. Error messages and logs
3. GitHub issues
4. Paper references for technical details