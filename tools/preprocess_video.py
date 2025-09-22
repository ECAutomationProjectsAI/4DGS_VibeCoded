#!/usr/bin/env python3
"""Preprocess video sequences for 4D Gaussian Splatting training.

Supports:
- Single video from monocular camera
- Multiple synchronized videos from different viewpoints
- Directory of videos (automatic multi-camera setup)
- Various video formats (mp4, avi, mov, mkv, etc.)

Usage examples:
    # Single video
    python preprocess_video.py video.mp4 output_dir/
    
    # Multiple camera videos
    python preprocess_video.py cam0.mp4 cam1.mp4 cam2.mp4 -o output_dir/
    
    # Directory of videos
    python preprocess_video.py video_folder/ -o output_dir/
    
    # With calibration
    python preprocess_video.py video.mp4 -o output_dir/ --calibration calib.json
    
    # Extract subset and resize
    python preprocess_video.py video.mp4 -o output_dir/ --start 10 --end 30 --resize 960 540
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gs4d.video_processor import (
    VideoProcessor,
    CameraConfig,
    VideoDataset,
    process_video_sequence
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_calibration(calib_arg: str) -> dict:
"""Preprocess video sequences for 4D Gaussian Splatting training.

Only accepts a directory path containing videos. All videos in the folder are processed in sorted order.
Each camera is named automatically from the video filename (without extension).

Supported formats: mp4, mov, avi, mkv.

Usage examples (RunPod/Ubuntu):
    # Directory of videos
    python3 tools/preprocess_video.py /workspace/videos -o /workspace/dataset \
        --resize 1280 720 --extract-every 1

    # With frame range
    python3 tools/preprocess_video.py /workspace/videos -o /workspace/dataset \
        --start_frame 0 --end_frame 1000
"""

def parse_calibration(calib_arg: str) -> dict:
    """Parse calibration argument (JSON file or string)."""
    if os.path.exists(calib_arg):
        with open(calib_arg, 'r') as f:
            return json.load(f)
    else:
        try:
            return json.loads(calib_arg)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse calibration: {calib_arg}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess a folder of videos for 4D Gaussian Splatting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (folder-only):
  # Directory of videos
  python3 tools/preprocess_video.py /workspace/videos -o /workspace/dataset \
      --resize 1280 720 --extract-every 1

Output structure:
  dataset/
    ├── frames/           # Extracted frames
    │   ├── <video_stem>/ # One subfolder per video (camera)
    │   └── ...
    ├── transforms.json   # Ready for training
    └── metadata.json     # Processing metadata
        """
    )

    parser.add_argument('input_dir', help='Input directory containing video files')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for processed data')

    # Strict frame index selection
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Start frame index (inclusive)')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='End frame index (exclusive)')
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                        help='Resize frames to specified dimensions')

    # Optional camera calibration mapping (by video stem name)
    parser.add_argument('--calibration', type=str,
                        help='Camera calibration JSON file or string (keys are video stem names)')

    # Processing options
    parser.add_argument('--extract-every', type=int, default=1,
                        help='Extract every N-th frame (default: 1 = every frame)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input directory
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        logger.error(f"Input must be a directory of videos: {input_dir}")
        sys.exit(1)

    logger.info(f"Processing directory: {input_dir}")

    # Parse calibration
    calibration = None
    if args.calibration:
        calibration = parse_calibration(args.calibration)

    # Create processor
    processor = VideoProcessor(
        output_dir=args.output,
        target_fps=30,
        resize=tuple(args.resize) if args.resize else None,
        extract_every_n=args.extract_every,
        use_gpu=args.use_gpu
    )

    # Discover videos in directory
    videos = sorted([
        str(Path(input_dir) / p)
        for p in os.listdir(input_dir)
        if p.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
    ])
    if len(videos) == 0:
        logger.error(f"No videos found in directory: {input_dir}")
        sys.exit(1)

    # Build camera configs with names from filenames
    cameras = []
    for vp in videos:
        stem = Path(vp).stem
        cam_name = stem.replace(' ', '_')
        cameras.append(CameraConfig(
            name=cam_name,
            video_path=vp,
            calibration=(calibration.get(cam_name) if (calibration and isinstance(calibration, dict)) else None),
            start_frame=args.start_frame,
            end_frame=args.end_frame
        ))

    # Process multi-camera (folder)
    metadata = processor.process_multi_camera_videos(cameras, sync_method='frame')

    # Create transforms.json for training
    transforms = processor.create_transforms_json(metadata)
    logger.info(f"Created transforms.json with {len(transforms['frames'])} frames")

    # Print summary
    print("\n" + "="*50)
    print("VIDEO PREPROCESSING COMPLETE")
    print("="*50)
    print(f"Output directory: {args.output}")
    print(f"Total frames extracted: {len(transforms['frames'])}")

    if 'cameras' in metadata:
        print(f"Number of cameras: {len(metadata['cameras'])}")
        for cam_name, cam_info in metadata['cameras'].items():
            print(f"  - {cam_name}: {cam_info['resolution'][0]}x{cam_info['resolution'][1]} @ {cam_info['fps']:.1f} fps")

    # Report failed videos if any
    if metadata.get('failed_videos'):
        print("\n" + "="*50)
        print("WARNING: Some videos failed to process:")
        print("="*50)
        for failed in metadata['failed_videos']:
            print(f"  ✗ {failed['camera']}: {failed['path']}")
            print(f"    Reason: {failed['error'][:100]}...")  # Truncate long errors
        print("="*50)
        print(f"Successfully processed: {len(metadata.get('cameras', {}))} videos")
        print(f"Failed: {len(metadata['failed_videos'])} videos")
        print("Continuing with available videos...")
        print("="*50)

    # Print time range
    times = [f['time'] for f in transforms['frames']]
    if times:
        print(f"\nTime range: {min(times):.2f} - {max(times):.2f} seconds")

    if len(transforms['frames']) > 0:
        print("\nReady for training with:")
        print(f"  python3 tools/train.py --data_root {args.output}")
    else:
        print("\nWARNING: No frames were extracted. Check your input videos.")
    print("="*50)
    main()