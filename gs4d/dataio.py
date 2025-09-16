import json
import os
import math
import torch
import imageio.v2 as imageio
import numpy as np
from typing import List, Dict, Optional, Tuple


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


def load_sequence(root: str, time_norm: bool = True, mask_root: Optional[str] = None, max_frames: int = -1, max_memory_gb: float = 8.0) -> Tuple[torch.Tensor, list, torch.Tensor, Optional[torch.Tensor]]:
    """
    Expect structure:
    root/
      frames/
        00000.png (or jpg)
      transforms.json with fields:
        frames: [ {file_path, transform_matrix (c2w)}, ... ]
        fl_x, fl_y, cx, cy, h, w
    
    Args:
        root: Dataset root directory
        time_norm: Whether to normalize timestamps
        mask_root: Optional mask directory
        max_frames: Maximum number of frames to load (-1 for all)
        max_memory_gb: Maximum memory to use for images (in GB)
        
    Returns: images [F,3,H,W], cams list, times [F,1], masks [F,1,H,W] or None
    """
    meta = load_transforms(os.path.join(root, 'transforms.json'))
    H, W = int(meta['h']), int(meta['w'])
    fx, fy = float(meta['fl_x']), float(meta['fl_y'])
    cx, cy = float(meta['cx']), float(meta['cy'])

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    frames = meta['frames']
    
    # Calculate memory usage per frame
    bytes_per_pixel = 4  # float32
    channels = 3
    memory_per_frame_gb = (H * W * channels * bytes_per_pixel) / (1024**3)
    total_frames = len(frames)
    
    # Check memory and limit frames if needed
    if max_frames > 0:
        frames = frames[:max_frames]
        print(f"  Limiting to {max_frames} frames (from {total_frames} total)")
    
    expected_memory_gb = len(frames) * memory_per_frame_gb
    if expected_memory_gb > max_memory_gb:
        max_allowed_frames = int(max_memory_gb / memory_per_frame_gb)
        print(f"  WARNING: Loading {len(frames)} frames would use {expected_memory_gb:.2f} GB")
        print(f"  Limiting to {max_allowed_frames} frames to stay under {max_memory_gb:.2f} GB")
        frames = frames[:max_allowed_frames]
    
    print(f"  Loading {len(frames)} frames ({len(frames) * memory_per_frame_gb:.2f} GB)...")
    
    images = []
    cams = []
    times = []
    masks = [] if mask_root is not None else None
    for i, fr in enumerate(frames):
        fp = os.path.join(root, fr['file_path'])
        im = imageio.imread(fp).astype(np.float32) / 255.0
        if im.shape[:2] != (H, W):
            raise ValueError(f"Image {fp} has shape {im.shape}, expected {(H,W)}")
        images.append(torch.from_numpy(im).permute(2, 0, 1))
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
