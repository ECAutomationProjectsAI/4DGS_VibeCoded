#!/usr/bin/env python3
"""
Step 1: Extract frames and map per-frame-per-camera groups for 4D Gaussian Splatting.

- Input: a folder of videos (each video is one camera)
- Output:
  dataset_root/
    extracted_frames/<cam_name>/*.jpg  # raw extracted frames per camera
    frames_mapped/frame000001/<cam_name>.jpg  # per-frame grouped views
    mapping.json  # summary of groups and cameras

Usage (RunPod/Ubuntu):
  python3 scripts/01_extract_and_map.py /workspace/videos -o /workspace/dataset \
      --resize 1280 720 --extract-every 1 --start_frame 0 --end_frame 900

Notes:
- Camera names are derived from video filenames (stems).
- Groups are created only for indices where all cameras have a frame.
- No COLMAP is run here. That is Step 2.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import logging

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gs4d.video_processor import VideoProcessor, CameraConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description='Step 1: Extract frames and build per-frame-per-camera mapping')
    ap.add_argument('input_dir', type=str, help='Folder containing video files (camera names derived from filenames)')
    ap.add_argument('--output', '-o', type=str, required=True, help='Dataset root for outputs')

    # Frame controls
    ap.add_argument('--start_frame', type=int, default=0, help='Start frame index (inclusive)')
    ap.add_argument('--end_frame', type=int, default=None, help='End frame index (exclusive)')
    ap.add_argument('--extract-every', type=int, default=1, help='Extract every Nth frame')

    # Resize and perf
    ap.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help='Resize frames to WIDTH HEIGHT')
    ap.add_argument('--use_gpu', action='store_true', help='Use GPU for video decode if available (OpenCV CUDA)')

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        logger.error(f"Input must be a directory of videos: {input_dir}")
        sys.exit(1)

    # Discover videos
    video_exts = ('.mp4', '.mov', '.avi', '.mkv')
    video_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in video_exts])
    if not video_files:
        logger.error(f"No videos found in {input_dir}")
        sys.exit(1)

    # Camera names from stems
    cameras: List[CameraConfig] = []
    for vf in video_files:
        cam_name = vf.stem.replace(' ', '_')
        cameras.append(CameraConfig(name=cam_name, video_path=str(vf)))
        logger.info(f"Detected camera: {cam_name} -> {vf.name}")

    # Extract frames
    processor = VideoProcessor(
        output_dir=str(out_root),
        target_fps=30.0,
        resize=tuple(args.resize) if args.resize else None,
        extract_every_n=int(args.extract_every),
        use_gpu=bool(args.use_gpu)
    )

    # Apply explicit frame range to all cameras
    for cam in cameras:
        cam.start_frame = int(args.start_frame)
        cam.end_frame = None if args.end_frame is None else int(args.end_frame)

    logger.info("\nExtracting frames per camera…")
    metadata = processor.process_multi_camera_videos(cameras, sync_method='frame')

    # Move frames/ -> extracted_frames/
    orig_frames_dir = out_root / 'frames'
    extracted_frames_dir = out_root / 'extracted_frames'
    extracted_frames_dir.mkdir(parents=True, exist_ok=True)

    if orig_frames_dir.exists():
        for cam_dir in orig_frames_dir.iterdir():
            if cam_dir.is_dir():
                dst = extracted_frames_dir / cam_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                for img in cam_dir.iterdir():
                    if img.is_file():
                        shutil.move(str(img), str(dst / img.name))
        # Cleanup empty directories
        try:
            for cam_dir in orig_frames_dir.iterdir():
                if cam_dir.is_dir():
                    cam_dir.rmdir()
            orig_frames_dir.rmdir()
        except Exception:
            pass

    # Update file paths in metadata to point to extracted_frames/
    for fr in metadata.get('frames', []):
        if fr.get('file_path', '').startswith('frames/'):
            fr['file_path'] = fr['file_path'].replace('frames/', 'extracted_frames/', 1)

    # Build frames_mapped/ with per-frame groups
    logger.info("\nBuilding per-frame-per-camera mapping (frames_mapped)…")
    frames_mapped_dir = out_root / 'frames_mapped'
    frames_mapped_dir.mkdir(parents=True, exist_ok=True)

    # Index frames by absolute frame index per camera
    frames_by_idx: Dict[int, Dict[str, str]] = {}
    cams_set = set(metadata['cameras'].keys())

    for fr in metadata.get('frames', []):
        # Frame index recorded during extraction
        idx = int(fr.get('frame_idx', -1))
        if idx < 0:
            # Derive from filename suffix
            try:
                idx = int(Path(fr['file_path']).stem.split('_')[-1])
            except Exception:
                continue
        cam = fr.get('camera', None)
        if cam is None:
            continue
        frames_by_idx.setdefault(idx, {})[cam] = fr['file_path']

    # Only keep indices where all cameras have an image
    ordered_indices = sorted([i for i, cmap in frames_by_idx.items() if cams_set.issubset(set(cmap.keys()))])
    if not ordered_indices:
        logger.error("No common frame indices across all cameras. Ensure videos are aligned or adjust --start/--end/--extract-every.")
        sys.exit(1)

    # Materialize mapping folders and record mapping
    mapping = {
        'groups': [],
        'camera_names': sorted(list(cams_set)),
        'source': 'extracted_frames',
    }

    for gi, idx in enumerate(ordered_indices, start=1):
        group_name = f"frame{gi:06d}"
        group_dir = frames_mapped_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        group_entry = {'name': group_name, 'index': idx, 'images': {}}
        for cam in sorted(cams_set):
            src_rel = frames_by_idx[idx][cam]
            src_abs = out_root / src_rel
            dst_rel = f"frames_mapped/{group_name}/{cam}.jpg"
            dst_abs = out_root / dst_rel
            if not src_abs.exists():
                logger.warning(f"Missing source image: {src_abs}")
                continue
            shutil.copy2(src_abs, dst_abs)
            group_entry['images'][cam] = dst_rel
        mapping['groups'].append(group_entry)

    # Save mapping summary and updated metadata
    with open(out_root / 'mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    with open(out_root / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Final report
    print("\n" + "="*60)
    print("✅ Step 1 Complete: Frames extracted and mapped")
    print("="*60)
    print(f"Cameras: {len(cams_set)} -> {', '.join(sorted(cams_set))}")
    print(f"Mapped groups: {len(mapping['groups'])}")
    print(f"Output root: {out_root}")
    print("\nNext: Run camera calibration (COLMAP) on the first mapped frame:")
    print(f"  python3 scripts/02_calibrate_cameras.py --data_root {out_root}")


if __name__ == '__main__':
    main()
