import json
import os
import math
import torch
import imageio.v2 as imageio
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2


def load_transforms(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def pose_from_colmap(c2w: np.ndarray) -> (np.ndarray, np.ndarray):
    # c2w 4x4, return R, t for world->cam
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    # invert c2w
    Rcw = R.T
    tcw = -Rcw @ t
    return Rcw, tcw


def _maybe_load_mask(mask_path: str, H: int, W: int) -> Optional[torch.Tensor]:
    if not os.path.isfile(mask_path):
        return None
    mk = imageio.imread(mask_path)
    if mk.ndim == 3:
        mk = mk[..., 0]
    if mk.shape != (H, W):
        return None
    mk = (mk.astype(np.float32) / 255.0)
    mk = torch.from_numpy(mk)[None, ...]  # [1,H,W]
    return mk


def load_sequence(root: str,
                  time_norm: bool = True,
                  mask_root: Optional[str] = None,
                  max_frames: int = -1,
                  max_memory_gb: float = -1,
                  start_frame: Optional[int] = None,
                  end_frame: Optional[int] = None) -> Tuple[torch.Tensor, list, torch.Tensor, Optional[torch.Tensor]]:
    """
    Expect structure:
    root/
      frames_mapped/frame000001/camA.jpg ...
      mapping.json (groups with per-frame-per-camera images)
      transforms.json with fields:
        Either:
          - frames: [ {file_path, transform_matrix (c2w)}, ... ]  (legacy)
        Or (compact, preferred):
          - cameras: { cam_name: { transform_matrix } }
          - intrinsics: fl_x, fl_y, cx, cy, h, w at top-level
    
    Args:
        root: Dataset root directory
        time_norm: Whether to normalize timestamps
        mask_root: Optional mask directory
        max_frames: Maximum number of frames to load (-1 for auto)
        max_memory_gb: Maximum memory to use for images (-1 for auto-detect)
        
    Returns: images [F,3,H,W], cams list, times [F,1], masks [F,1,H,W] or None
    """
    # Auto-detect memory if needed
    if max_memory_gb == -1:
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            # Use 90% of available memory by default (aggressive)
            max_memory_gb = available_gb * 0.90
            print(f"  🔧 Auto-detected memory limit: {max_memory_gb:.1f} GB (90% of {available_gb:.1f} GB available)")
        except ImportError:
            # Fallback if psutil not available
            max_memory_gb = 8.0
            print(f"  ⚠️  Using default memory limit: {max_memory_gb:.1f} GB (install psutil for auto-detection)")
    
    meta = load_transforms(os.path.join(root, 'transforms.json'))
    H, W = int(meta['h']), int(meta['w'])
    fx, fy = float(meta['fl_x']), float(meta['fl_y'])
    cx, cy = float(meta['cx']), float(meta['cy'])

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Build frames list: prefer compact (mapping.json + per-camera pose) if available
    frames = []
    mapping_path = os.path.join(root, 'mapping.json')
    if os.path.isfile(mapping_path) and 'cameras' in meta and ('frames' not in meta or len(meta.get('frames', [])) == 0):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        cam_poses = {k: np.array(v['transform_matrix'], dtype=np.float32) for k, v in meta['cameras'].items()}
        groups = mapping.get('groups', [])
        # Build per-image entries from mapping using per-camera pose
        fps = 30.0  # nominal, only used for relative times and normalized later
        for gi, group in enumerate(groups, start=1):
            abs_idx = int(group.get('index', gi))
            t = abs_idx / fps
            images = group.get('images', {})
            for cam, rel_path in images.items():
                if cam not in cam_poses:
                    continue
                frames.append({
                    'file_path': rel_path,
                    'transform_matrix': cam_poses[cam].tolist(),
                    'time': t,
                    'camera': cam,
                    'frame_idx': abs_idx
                })
    else:
        # Legacy path: use frames array from transforms.json
        frames = meta['frames']

    # Strict frame range filtering if requested
    if start_frame is not None or end_frame is not None:
        sf = 0 if start_frame is None else int(start_frame)
        ef = int(1e18) if end_frame is None else int(end_frame)
        def in_range(fr, idx):
            fid = fr.get('frame_idx', idx)
            return sf <= fid < ef
        frames = [fr for idx, fr in enumerate(frames) if in_range(fr, idx)]
    
    # Calculate memory usage per frame
    bytes_per_pixel = 4  # float32
    channels = 3
    memory_per_frame_gb = (H * W * channels * bytes_per_pixel) / (1024**3)
    total_frames = len(frames)
    
    # Auto-detect max frames based on memory if not specified
    if max_frames == -1:
        max_frames = int(max_memory_gb / memory_per_frame_gb)
        print(f"  Auto-calculated max frames: {max_frames} (based on {max_memory_gb:.1f} GB limit)")
    
    # Apply frame limits
    if max_frames > 0 and max_frames < len(frames):
        frames = frames[:max_frames]
        print(f"  Loading {max_frames} frames (from {total_frames} total)")
    
    expected_memory_gb = len(frames) * memory_per_frame_gb
    if expected_memory_gb > max_memory_gb:
        max_allowed_frames = int(max_memory_gb / memory_per_frame_gb)
        print(f"  WARNING: Loading {len(frames)} frames would use {expected_memory_gb:.2f} GB")
        print(f"  Limiting to {max_allowed_frames} frames to stay under {max_memory_gb:.2f} GB")
        frames = frames[:max_allowed_frames]
    
    print(f"  📁 Loading {len(frames)} frames ({len(frames) * memory_per_frame_gb:.2f} GB estimated)...")
    
    # Check if we have enough memory before starting
    try:
        import psutil
        mem = psutil.virtual_memory()
        current_available_gb = mem.available / (1024**3)
        if len(frames) * memory_per_frame_gb > current_available_gb:
            print(f"\n❌ ERROR: Not enough memory to load frames!")
            print(f"   Need: {len(frames) * memory_per_frame_gb:.2f} GB")
            print(f"   Available: {current_available_gb:.2f} GB")
            print(f"\n💡 Solutions:")
            print(f"   1. Reduce frames with --max_frames {int(current_available_gb / memory_per_frame_gb * 0.8)}")
            print(f"   2. Reduce resolution in preprocessing")
            print(f"   3. Close other applications")
            raise MemoryError(f"Insufficient memory: need {len(frames) * memory_per_frame_gb:.2f}GB, have {current_available_gb:.2f}GB")
    except ImportError:
        pass  # psutil not available, skip check
    
    from tqdm import tqdm
    
    images = []
    cams = []
    times = []
    masks = [] if mask_root is not None else None
    
    # Progress bar for loading frames
    pbar = tqdm(enumerate(frames), total=len(frames), desc="Loading frames", unit="frame")
    for i, fr in pbar:
        fp = os.path.join(root, fr['file_path'])
        
        # Check if file exists
        if not os.path.exists(fp):
            print(f"\n❌ ERROR: Frame file not found: {fp}")
            raise FileNotFoundError(f"Frame file not found: {fp}")
        
        im = imageio.imread(fp)
        # Ensure channel dimension is 3 (RGB)
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=2)
        if im.shape[-1] == 4:
            im = im[..., :3]

        oh, ow = im.shape[:2]
        if (oh, ow) != (H, W):
            # Resize to expected resolution from transforms.json
            # Choose interpolation based on scaling direction
            interp = cv2.INTER_AREA if (H < oh or W < ow) else cv2.INTER_LINEAR
            im = cv2.resize(im, (W, H), interpolation=interp)
        im = im.astype(np.float32) / 255.0
        images.append(torch.from_numpy(im).permute(2, 0, 1))
        
        # Update progress bar with memory usage
        if i % 10 == 0:  # Update every 10 frames
            loaded_gb = (i + 1) * memory_per_frame_gb
            pbar.set_postfix({'Loaded': f'{loaded_gb:.1f}GB', 'Progress': f'{(i+1)/len(frames)*100:.0f}%'})
        c2w = np.array(fr['transform_matrix'], dtype=np.float32)
        R, t = pose_from_colmap(c2w)
        cams.append({
            'K': torch.from_numpy(K),
            'R': torch.from_numpy(R),
            't': torch.from_numpy(t),
            'H': H,
            'W': W,
        })
        times.append(fr.get('time', i))
        if masks is not None:
            # heuristic: find mask by basename under mask_root
            bname = os.path.basename(fr['file_path'])
            candidate = os.path.join(mask_root, bname)
            mk = _maybe_load_mask(candidate, H, W)
            if mk is None:
                # also try replacing 'frames' with 'masks' in path
                alt = os.path.join(mask_root, os.path.basename(bname))
                mk = _maybe_load_mask(alt, H, W)
            masks.append(mk if mk is not None else torch.zeros(1, H, W))
    images = torch.stack(images, dim=0)  # [F,3,H,W]
    times = np.array(times, dtype=np.float32)
    if time_norm:
        tmin, tmax = times.min(), times.max() if len(times) > 1 else (0, 1)
        denom = (tmax - tmin) if (tmax - tmin) > 1e-6 else 1.0
        times = (times - tmin) / denom - 0.5
    times = torch.from_numpy(times)[:, None]
    if masks is not None:
        masks = torch.stack(masks, dim=0)  # [F,1,H,W]
    return images, cams, times, masks
