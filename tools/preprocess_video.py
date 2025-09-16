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
        description='Preprocess videos for 4D Gaussian Splatting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video (monocular)
  python preprocess_video.py video.mp4 -o dataset/

  # Multiple synchronized cameras
  python preprocess_video.py cam0.mp4 cam1.mp4 cam2.mp4 -o dataset/

  # Directory of videos
  python preprocess_video.py videos/ -o dataset/

  # With time range and resize
  python preprocess_video.py video.mp4 -o dataset/ --start 5.0 --end 15.0 --resize 1280 720

  # Extract every 5th frame for faster processing
  python preprocess_video.py video.mp4 -o dataset/ --extract-every 5

  # With camera calibration
  python preprocess_video.py video.mp4 -o dataset/ --calibration calibration.json

  # For COLMAP reconstruction
  python preprocess_video.py video.mp4 -o dataset/ --colmap

Output structure:
  dataset/
    ├── frames/           # Extracted frames
    │   ├── cam0/        # Camera 0 frames
    │   ├── cam1/        # Camera 1 frames (if multi-camera)
    │   └── ...
    ├── transforms.json   # Ready for training
    └── metadata.json     # Processing metadata
        """
    )
    
    parser.add_argument('input', nargs='+', 
                       help='Input video file(s) or directory')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for processed data')
    
    # Video processing options
    parser.add_argument('--start', type=float, default=0.0,
                       help='Start time in seconds')
    parser.add_argument('--end', type=float, default=None,
                       help='End time in seconds')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Target FPS for extraction (default: 30)')
    parser.add_argument('--extract-every', type=int, default=1,
                       help='Extract every N frames (default: 1)')
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='Resize frames to specified dimensions')
    
    # Camera options
    parser.add_argument('--calibration', type=str,
                       help='Camera calibration JSON file or string')
    parser.add_argument('--camera-names', nargs='+',
                       help='Names for cameras (default: cam0, cam1, ...)')
    parser.add_argument('--sync-offsets', nargs='+', type=float,
                       help='Time offsets for camera synchronization')
    
    # COLMAP options
    parser.add_argument('--colmap', action='store_true',
                       help='Also extract in COLMAP format for SfM')
    parser.add_argument('--colmap-dir', type=str,
                       help='Output directory for COLMAP (default: output/colmap)')
    
    # Processing options
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration if available')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine input type
    input_paths = args.input
    
    if len(input_paths) == 1 and os.path.isdir(input_paths[0]):
        # Directory of videos
        logger.info(f"Processing directory: {input_paths[0]}")
        input_path = input_paths[0]
        
    elif len(input_paths) == 1:
        # Single video
        logger.info(f"Processing single video: {input_paths[0]}")
        input_path = input_paths[0]
        
    else:
        # Multiple videos
        logger.info(f"Processing {len(input_paths)} videos")
        input_path = input_paths
    
    # Parse calibration
    calibration = None
    if args.calibration:
        calibration = parse_calibration(args.calibration)
    
    # Create processor
    processor = VideoProcessor(
        output_dir=args.output,
        target_fps=args.fps,
        resize=tuple(args.resize) if args.resize else None,
        extract_every_n=args.extract_every,
        use_gpu=args.use_gpu
    )
    
    # Handle different input types
    if isinstance(input_path, list):
        # Multiple videos - create camera configs
        cameras = []
        for idx, video_path in enumerate(input_path):
            if not os.path.exists(video_path):
                logger.error(f"Video not found: {video_path}")
                sys.exit(1)
            
            cam_name = f"cam{idx}"
            if args.camera_names and idx < len(args.camera_names):
                cam_name = args.camera_names[idx]
            
            sync_offset = 0.0
            if args.sync_offsets and idx < len(args.sync_offsets):
                sync_offset = args.sync_offsets[idx]
            
            cameras.append(CameraConfig(
                name=cam_name,
                video_path=video_path,
                calibration=calibration.get(cam_name) if calibration else None,
                sync_offset=sync_offset
            ))
        
        # Process multi-camera
        metadata = processor.process_multi_camera_videos(cameras)
        
    elif os.path.isdir(input_path):
        # Directory of videos
        metadata = processor.process_video_directory(
            input_path,
            calibration_file=args.calibration
        )
        
    else:
        # Single video
        if not os.path.exists(input_path):
            logger.error(f"Video not found: {input_path}")
            sys.exit(1)
        
        metadata = processor.process_single_camera_video(
            input_path,
            camera_name=args.camera_names[0] if args.camera_names else "cam0",
            calibration=calibration,
            start_time=args.start,
            end_time=args.end
        )
    
    # Create transforms.json for training
    transforms = processor.create_transforms_json(metadata)
    logger.info(f"Created transforms.json with {len(transforms['frames'])} frames")
    
    # Extract for COLMAP if requested
    if args.colmap:
        colmap_dir = args.colmap_dir or os.path.join(args.output, "colmap")
        
        if isinstance(input_path, str) and not os.path.isdir(input_path):
            # Single video - can extract for COLMAP
            logger.info(f"Extracting frames for COLMAP to {colmap_dir}")
            processor.extract_colmap_format(input_path, colmap_dir)
        else:
            logger.warning("COLMAP extraction only supported for single video")
    
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
    
    # Print time range
    times = [f['time'] for f in transforms['frames']]
    if times:
        print(f"Time range: {min(times):.2f} - {max(times):.2f} seconds")
    
    print("\nReady for training with:")
    print(f"  python train.py --data_root {args.output}")
    print("="*50)


if __name__ == '__main__':
    main()