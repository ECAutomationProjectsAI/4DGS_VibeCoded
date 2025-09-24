#!/usr/bin/env python3
"""
Step 2: Camera calibration using COLMAP (first mapped frame only) and transforms.json generation.

This script:
- Uses only images from frames_mapped/frame000001 (one per camera) to compute static
  intrinsics/extrinsics via COLMAP. This avoids dynamic-scene failure.
- Applies the computed per-camera extrinsics to ALL mapped frames.
- Writes a transforms.json that references frames_mapped/* and is compatible with gs4d/dataio.py.

Usage (RunPod/Ubuntu):
  python3 scripts/02_calibrate_cameras.py --data_root /workspace/dataset \
      --camera_model OPENCV --threads 8

Outputs:
  data_root/
    colmap/                 # workspace with logs and outputs
    transforms.json         # final dataset transforms (uses frames_mapped/*)
    colmap_parsed.json      # parsed COLMAP cameras/images (debug)

Requirements:
- COLMAP must be installed and on PATH inside the container.
- Run this after scripts/01_extract_and_map.py.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import shutil
import numpy as np
import logging

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gs4d.colmap_utils import run_colmap_sfm, parse_colmap_output, quaternion_to_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def main():
    ap = argparse.ArgumentParser(description='Step 2: Calibrate cameras (COLMAP first-frame) and create transforms.json')
    ap.add_argument('--data_root', type=str, required=True, help='Dataset root from Step 1')
    ap.add_argument('--camera_model', type=str, default='OPENCV', choices=['OPENCV','PINHOLE','SIMPLE_RADIAL','OPENCV_FISHEYE'], help='COLMAP camera model')
    ap.add_argument('--threads', type=int, default=8, help='SIFT threads for COLMAP')
    ap.add_argument('--max_image_size', type=int, default=0, help='Downscale images for SIFT (0 = original)')
    ap.add_argument('--max_num_features', type=int, default=0, help='Limit SIFT features per image (0 = default)')

    args = ap.parse_args()

    root = Path(args.data_root)
    frames_mapped = root / 'frames_mapped'
    first_group = frames_mapped / 'frame000001'
    if not first_group.exists():
        logger.error(f"First mapped frame not found: {first_group}. Run Step 1 first.")
        sys.exit(1)

    # Prepare COLMAP images folder with one image per camera from first group
    colmap_root = root / 'colmap'
    images_dir = colmap_root / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous images directory to avoid mixing old runs
    for old in images_dir.glob('*'):
        try:
            old.unlink()
        except Exception:
            pass

    copied = 0
    for img in sorted(first_group.glob('*.jpg')):
        cam_name = img.stem  # "camXX"
        dst = images_dir / f"{cam_name}.jpg"
        shutil.copy2(img, dst)
        copied += 1
    if copied == 0:
        logger.error(f"No images found in {first_group}. Ensure Step 1 created frames_mapped correctly.")
        sys.exit(1)
    logger.info(f"Prepared {copied} images for COLMAP from {first_group}")

    # Run COLMAP via utility module
    ok = run_colmap_sfm(
        image_dir=str(images_dir),
        colmap_dir=str(colmap_root),
        camera_model=args.camera_model,
        threads=args.threads,
        max_image_size=args.max_image_size,
        max_num_features=args.max_num_features,
        matcher='exhaustive',
        single_camera=False,
    )
    if not ok:
        logger.error("COLMAP failed. Check logs under data_root/colmap/logs/*.log. You can re-run with simpler model: --camera_model PINHOLE or use --max_image_size 1600")
        sys.exit(1)

    colmap_data = parse_colmap_output(str(colmap_root))
    with open(colmap_root / 'colmap_parsed.json', 'w') as f:
        json.dump(colmap_data, f, indent=2)

    # Build per-camera pose map c2w from images.txt
    cam_pose_map = {}
    for image_name, img_data in colmap_data.get('images', {}).items():
        # image_name is like "camXX.jpg"
        cam = Path(image_name).stem
        qw, qx, qy, qz = map(float, img_data['quaternion'])
        tx, ty, tz = map(float, img_data['translation'])
        R = quaternion_to_matrix(qw, qx, qy, qz)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ np.array([tx, ty, tz], dtype=np.float32)
        cam_pose_map[cam] = c2w

    if not cam_pose_map:
        logger.error("No camera poses parsed from COLMAP. Inspect colmap/sparse_txt/images.txt")
        sys.exit(1)

    # Choose global intrinsics (from first camera). All images were resized in Step 1 to be consistent.
    intr = None
    if colmap_data['cameras']:
        intr = list(colmap_data['cameras'].values())[0]
    else:
        logger.error("No camera intrinsics found in COLMAP outputs.")
        sys.exit(1)

    width = int(intr['width']); height = int(intr['height'])
    params = intr['params']
    if intr['model'] in ['OPENCV','PINHOLE']:
        fx = float(params[0]); fy = float(params[1] if len(params) > 1 else params[0])
        cx = float(params[2] if len(params) > 2 else width/2)
        cy = float(params[3] if len(params) > 3 else height/2)
    else:
        # Fallback sensible defaults
        fx = max(width, height); fy = fx; cx = width/2; cy = height/2

    # Load mapping.json to enumerate all frames
    mapping_path = root / 'mapping.json'
    if not mapping_path.exists():
        logger.error(f"mapping.json not found at {mapping_path}. Run Step 1 first.")
        sys.exit(1)
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    camera_names = mapping.get('camera_names', [])
    if not camera_names:
        # derive from cam_pose_map
        camera_names = sorted(list(cam_pose_map.keys()))

    # Compose compact transforms.json: store intrinsics once and per-camera pose only.
    # Training/data loader will combine this with mapping.json to enumerate all frames.
    transforms = {
        'camera_angle_x': 2.0 * float(np.arctan(width / (2.0 * fx))),
        'fl_x': fx,
        'fl_y': fy,
        'cx': cx,
        'cy': cy,
        'w': width,
        'h': height,
        'aabb_scale': 4,
        'scale': 1.0,
        'offset': [0.5, 0.5, 0.5],
        'cameras': {}
    }

    # Save per-camera c2w from COLMAP first-frame calibration
    for cam in sorted(camera_names):
        c2w = cam_pose_map.get(cam)
        if c2w is None:
            continue
        transforms['cameras'][cam] = {
            'transform_matrix': c2w.tolist()
        }

    out_path = root / 'transforms.json'
    with open(out_path, 'w') as f:
        json.dump(transforms, f, indent=2)

    print("\n" + "="*60)
    print("✅ Step 2 Complete: Camera calibration + transforms.json")
    print("="*60)
    print(f"Cameras calibrated: {len(cam_pose_map)} -> {', '.join(sorted(cam_pose_map.keys()))}")
    print(f"Frames in transforms: {len(transforms['frames'])}")
    print(f"Output: {out_path}")
    print("\nNext: Train the 4D model:")
    print(f"  python3 scripts/03_train_4dgs.py --data_root {root} --out_dir {root / 'outputs'}")


if __name__ == '__main__':
    main()
