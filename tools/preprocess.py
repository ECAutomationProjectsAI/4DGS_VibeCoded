#!/usr/bin/env python3
"""
Folder-only multiview preprocessing for 4D Gaussian Splatting.

- Input: a directory containing multiple videos (mp4, mov, avi, mkv)
- Cameras are auto-named using each video filename (without extension)
- Outputs frames/ per camera and a transforms.json suitable for training
- Optional: run COLMAP to estimate intrinsics/extrinsics

Usage (RunPod/Ubuntu):
    python3 tools/preprocess.py /workspace/videos -o /workspace/dataset \
        --resize 1920 1080 --extract-every 1
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gs4d.video_processor import VideoProcessor, CameraConfig

# Reuse multiview pipeline
from tools.preprocess_multiview import MultiViewPreprocessor  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_calibration(calib_arg: Optional[str]) -> Optional[Dict]:
    if not calib_arg:
        return None
    if os.path.exists(calib_arg):
        with open(calib_arg, 'r') as f:
            return json.load(f)
    try:
        return json.loads(calib_arg)
    except Exception:
        logger.warning(f"Could not parse calibration: {calib_arg}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess multi-view videos (folder-only) for 4D Gaussian Splatting',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='Example: python3 tools/preprocess.py /workspace/videos -o /workspace/dataset --resize 1920 1080 --extract-every 1'
    )

    parser.add_argument('input_dir', type=str, help='Folder containing video files (camera names derived from filenames)')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory')
    parser.add_argument('--calibration', type=str, default=None, help='Calibration JSON file or JSON string (optional)')

    # Frame control
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame index (inclusive)')
    parser.add_argument('--end_frame', type=int, default=None, help='End frame index (exclusive)')
    parser.add_argument('--extract-every', type=int, default=1, help='Extract every Nth frame (default: 1)')

    # Resize and perf
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help='Resize frames to WIDTH HEIGHT')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for preprocessing if available (not for COLMAP)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    # COLMAP controls
    parser.add_argument('--skip_colmap', action='store_true', help='Skip running COLMAP')
    parser.add_argument('--colmap_mapped_groups', type=int, default=3, help='Groups for COLMAP subset (default: 3)')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        logger.error(f"Input must be a directory of videos: {input_dir}")
        sys.exit(1)

    # Discover videos
    exts = ('.mp4', '.mov', '.avi', '.mkv')
    videos = sorted([str(Path(input_dir) / p) for p in os.listdir(input_dir) if p.lower().endswith(exts)])
    logger.info(f"Scanning directory: {input_dir}")
    logger.info(f"Found {len(videos)} video file(s)")
    for v in videos:
        logger.info(f"  - {os.path.basename(v)}")
    if not videos:
        logger.error("No videos found to process.")
        sys.exit(1)

    # Initialize preprocessor (from multiview pipeline)
    pre = MultiViewPreprocessor(
        output_dir=args.output,
        use_gpu=args.use_gpu,
        skip_colmap=args.skip_colmap,
        colmap_mapped_groups=args.colmap_mapped_groups
    )

    # Pass through to multiview folder runner
    resize_tuple = tuple(args.resize) if args.resize else None
    result = pre.process_video_folder(
        video_folder=input_dir,
        output_dir=args.output,
        fps=30,
        skip_frames=args.extract_every,
        resize=resize_tuple,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )

    if result.get('success'):
        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Cameras processed: {result['num_cameras']}")
        print(f"Total frames: {result['total_frames']}")
        print(f"Output directory: {result['output_dir']}")
        print("\nNext step:")
        print(f"  python tools/train.py --data_root {result['output_dir']} --out_dir model/")
        sys.exit(0)
    else:
        logger.error("Preprocessing failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
