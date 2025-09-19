#!/usr/bin/env python
"""
Complete preprocessing pipeline for 4D Gaussian Splatting.

This script handles the full data preparation workflow:
1. Extract frames from multi-view videos (from folder or individual files)
2. Camera calibration (intrinsics/extrinsics) via COLMAP or provided calibration
3. Time synchronization across cameras
4. Generate transforms.json for training
5. Optional: mask generation, depth estimation

Based on techniques from SpacetimeGaussian, 4K4D, and FreeTimeGS papers.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import subprocess
import shutil
from tqdm import tqdm
import logging
import glob

# Add parent directory to import gs4d modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs4d.video_processor import VideoProcessor, CameraConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiViewPreprocessor:
    """Complete preprocessing pipeline for 4DGS."""
    
    def __init__(self, output_dir: str, use_gpu: bool = False, skip_colmap: bool = False):
        """
        Initialize preprocessor.
        
        Args:
            output_dir: Output directory for processed data
            use_gpu: Use GPU acceleration if available
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        self.skip_colmap = skip_colmap
        
        # Subdirectories
        self.frames_dir = self.output_dir / "frames"
        self.colmap_dir = self.output_dir / "colmap"
        self.masks_dir = self.output_dir / "masks"
        
    def run_colmap_sfm(self, 
                       image_dir: str,
                       camera_model: str = "OPENCV",
                       single_camera: bool = False,
                       gpu_index: int = 0) -> bool:
        """
        Run COLMAP Structure-from-Motion pipeline.
        
        Based on SpacetimeGaussian approach: use SfM across timestamps
        to get initial 3D points and camera poses.
        
        Args:
            image_dir: Directory containing images
            camera_model: Camera model (OPENCV, PINHOLE, etc.)
            single_camera: Whether all images are from same camera
            gpu_index: GPU index for feature extraction
            
        Returns:
            Success status
        """
        logger.info("Running COLMAP SfM pipeline...")
        
        # Create COLMAP workspace
        self.colmap_dir.mkdir(exist_ok=True)
        database_path = self.colmap_dir / "database.db"
        
        # Prepare a clean environment for headless COLMAP
        colmap_env = os.environ.copy()
        # Avoid cv2's Qt plugin path which can break COLMAP in headless environments
        colmap_env.pop('QT_PLUGIN_PATH', None)
        colmap_env.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
        # Force offscreen platform to avoid X11/xcb requirement
        colmap_env['QT_QPA_PLATFORM'] = 'offscreen'

        # Helper: check if a COLMAP subcommand supports an option
        def _colmap_has_option(subcmd: str, option: str) -> bool:
            try:
                help_proc = subprocess.run([
                    "colmap", subcmd, "-h"
                ], check=False, capture_output=True, text=True, env=colmap_env)
                return option in (help_proc.stdout or "")
            except Exception:
                return False

        # Step 1: Feature extraction
        logger.info("  1. Extracting features...")
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--ImageReader.camera_model", camera_model
        ]
        # Force CPU SIFT in headless environments if supported to avoid OpenGL context errors
        if _colmap_has_option("feature_extractor", "--SiftExtraction.use_gpu"):
            cmd.extend(["--SiftExtraction.use_gpu", "0"])  # disable GPU SIFT
        
        if single_camera:
            cmd.extend(["--ImageReader.single_camera", "1"])
        
        try:
            # Stream COLMAP output for visibility
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=colmap_env)
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if line:
                    logger.info(line)
            ret = proc.wait()
            if ret != 0:
                logger.error(f"Feature extraction failed with code {ret}")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Feature extraction failed: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("COLMAP not found! Please install COLMAP and add it to PATH")
            logger.info("Download from: https://github.com/colmap/colmap/releases")
            return False
        
        # Step 2: Feature matching
        # Choose matcher to avoid O(N^2) for large N
        num_images = len([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
        if num_images > 800:
            logger.info(f"  2. Matching features (sequential matcher, images={num_images})...")
            cmd = [
                "colmap", "sequential_matcher",
                "--database_path", str(database_path),
                "--SequentialMatching.overlap", "5"
            ]
        else:
            logger.info(f"  2. Matching features (exhaustive matcher, images={num_images})...")
            cmd = [
                "colmap", "exhaustive_matcher",
                "--database_path", str(database_path)
            ]
        # Force CPU matching if supported to avoid GPU dependencies
        if _colmap_has_option(cmd[1], "--SiftMatching.use_gpu"):
            cmd.extend(["--SiftMatching.use_gpu", "0"])  # disable GPU matching
        
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=colmap_env)
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if line:
                    logger.info(line)
            ret = proc.wait()
            if ret != 0:
                logger.error(f"Feature matching failed with code {ret}")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Feature matching failed: {e.stderr}")
            return False
        
        # Step 3: Sparse reconstruction
        logger.info("  3. Running sparse reconstruction...")
        sparse_dir = self.colmap_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        
        cmd = [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir)
        ]
        
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=colmap_env)
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if line:
                    logger.info(line)
            ret = proc.wait()
            if ret != 0:
                logger.error(f"Sparse reconstruction failed with code {ret}")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Sparse reconstruction failed: {e.stderr}")
            return False
        
        # Step 4: Convert to text format
        logger.info("  4. Converting to text format...")
        sparse_txt_dir = self.colmap_dir / "sparse_txt"
        sparse_txt_dir.mkdir(exist_ok=True)
        
        # Find the reconstruction folder (usually '0')
        recon_dirs = list(sparse_dir.glob("*"))
        if not recon_dirs:
            logger.error("No reconstruction found!")
            return False
        
        recon_dir = recon_dirs[0]  # Use first reconstruction
        
        cmd = [
            "colmap", "model_converter",
            "--input_path", str(recon_dir),
            "--output_path", str(sparse_txt_dir),
            "--output_type", "TXT"
        ]
        
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=colmap_env)
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if line:
                    logger.info(line)
            ret = proc.wait()
            if ret != 0:
                logger.error(f"Model conversion failed with code {ret}")
                return False
            logger.info("✅ COLMAP SfM completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Model conversion failed: {e.stderr}")
            return False
    
    def parse_colmap_output(self) -> Dict:
        """
        Parse COLMAP output to extract camera parameters and poses.
        
        Returns:
            Dictionary with cameras and images data
        """
        sparse_txt = self.colmap_dir / "sparse_txt"
        
        # Parse cameras.txt
        cameras = {}
        cameras_file = sparse_txt / "cameras.txt"
        if cameras_file.exists():
            with open(cameras_file, 'r') as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cam_id = int(parts[0])
                        model = parts[1]
                        width = int(parts[2])
                        height = int(parts[3])
                        params = list(map(float, parts[4:]))
                        
                        cameras[cam_id] = {
                            'model': model,
                            'width': width,
                            'height': height,
                            'params': params
                        }
        
        # Parse images.txt
        images = {}
        images_file = sparse_txt / "images.txt"
        if images_file.exists():
            with open(images_file, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 2):
                    if lines[i].startswith("#"):
                        continue
                    
                    parts = lines[i].strip().split()
                    if len(parts) >= 10:
                        image_id = int(parts[0])
                        qw, qx, qy, qz = map(float, parts[1:5])
                        tx, ty, tz = map(float, parts[5:8])
                        camera_id = int(parts[8])
                        image_name = parts[9]
                        
                        images[image_name] = {
                            'id': image_id,
                            'quaternion': [qw, qx, qy, qz],
                            'translation': [tx, ty, tz],
                            'camera_id': camera_id
                        }
        
        return {'cameras': cameras, 'images': images}
    
    def generate_transforms_json(self,
                                colmap_data: Dict,
                                frame_metadata: List[Dict]) -> Dict:
        """
        Generate transforms.json file compatible with 4DGS training.
        
        Incorporates techniques from multiple papers:
        - SpacetimeGaussian: temporal parameters
        - 4K4D: multi-view blending weights
        - FreeTimeGS: explicit time values
        
        Args:
            colmap_data: Parsed COLMAP output
            frame_metadata: Frame information from video processing
            
        Returns:
            Transforms dictionary
        """
        transforms = {
            'frames': [],
            'camera_angle_x': 0,  # Will be computed
            'fl_x': 0,
            'fl_y': 0,
            'cx': 0,
            'cy': 0,
            'w': 0,
            'h': 0,
            'aabb_scale': 4,  # Scene scale
            'scale': 1.0,
            'offset': [0.5, 0.5, 0.5]
        }
        
        # Get camera intrinsics (assume single camera or first camera)
        if colmap_data['cameras']:
            cam = list(colmap_data['cameras'].values())[0]
            transforms['w'] = cam['width']
            transforms['h'] = cam['height']
            
            if cam['model'] in ['OPENCV', 'PINHOLE']:
                # fx, fy, cx, cy, k1, k2, p1, p2
                transforms['fl_x'] = cam['params'][0]
                transforms['fl_y'] = cam['params'][1] if len(cam['params']) > 1 else cam['params'][0]
                transforms['cx'] = cam['params'][2] if len(cam['params']) > 2 else cam['width'] / 2
                transforms['cy'] = cam['params'][3] if len(cam['params']) > 3 else cam['height'] / 2
                
                # Camera angle from focal length
                transforms['camera_angle_x'] = 2 * np.arctan(cam['width'] / (2 * transforms['fl_x']))
        
        # Process each frame
        for frame_info in frame_metadata:
            file_path = frame_info['file_path']
            image_name = Path(file_path).name
            
            if image_name in colmap_data['images']:
                img_data = colmap_data['images'][image_name]
                
                # Convert quaternion and translation to 4x4 matrix
                qw, qx, qy, qz = img_data['quaternion']
                tx, ty, tz = img_data['translation']
                
                # Quaternion to rotation matrix
                R = quaternion_to_matrix(qw, qx, qy, qz)
                
                # Create c2w (camera-to-world) transform
                c2w = np.eye(4)
                c2w[:3, :3] = R.T  # Inverse rotation
                c2w[:3, 3] = -R.T @ np.array([tx, ty, tz])  # Camera position
                
                frame_data = {
                    'file_path': file_path,
                    'transform_matrix': c2w.tolist(),
                    'time': frame_info.get('time', 0.0),  # Temporal information
                    'camera': frame_info.get('camera', 'cam0')
                }
                
                transforms['frames'].append(frame_data)
        
        return transforms
    
    def process_video_folder(self, 
                           video_folder: Union[str, Path],
                           output_dir: Union[str, Path],
                           video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv'],
                           fps: int = 30,
                           skip_frames: int = 1,
                           resize: Optional[Tuple[int, int]] = None,
                           start_time: float = 0.0,
                           end_time: Optional[float] = None) -> Dict:
        """
        Process all videos in a folder.
        
        Args:
            video_folder: Path to folder containing video files
            output_dir: Output directory for processed data
            video_extensions: List of video file extensions to process
            fps: Frame rate for extraction
            skip_frames: Extract every N frames
            resize: Optional (width, height) to resize frames
            start_time: Start time in seconds
            end_time: End time in seconds (None for full video)
            
        Returns:
            Processing results dictionary
        """
        video_folder = Path(video_folder)
        output_dir = Path(output_dir)
        
        if not video_folder.exists():
            raise ValueError(f"Video folder does not exist: {video_folder}")
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_folder.glob(f"*{ext}"))
            video_files.extend(video_folder.glob(f"*{ext.upper()}"))
        
        video_files = sorted(list(set(video_files)))  # Remove duplicates and sort
        
        if not video_files:
            raise ValueError(f"No video files found in {video_folder} with extensions {video_extensions}")
        
        logger.info(f"Found {len(video_files)} video files in {video_folder}")
        
        # Use filenames (without extension) as camera names
        camera_names = [vf.stem for vf in video_files]
        
        logger.info("Videos to process:")
        for vf, cn in zip(video_files, camera_names):
            logger.info(f"  {vf.name} -> camera name: {cn}")
        
        # Initialize video processor for output directory
        from gs4d.video_processor import VideoProcessor
        processor = VideoProcessor(
            output_dir=str(self.output_dir),
            target_fps=fps,
            resize=resize,
            extract_every_n=skip_frames,
            use_gpu=self.use_gpu
        )
        
        # Process each video
        frame_metadata = []
        for video_file, camera_name in zip(video_files, camera_names):
            logger.info(f"\nProcessing {video_file.name} as camera '{camera_name}'...")
            
            # Process this video
            camera_metadata = processor.process_single_camera_video(
                video_path=str(video_file),
                camera_name=camera_name,
                start_time=start_time,
                end_time=end_time
            )
            
            # Check for errors
            if 'error' in camera_metadata:
                logger.error(f"Failed to process {video_file.name}: {camera_metadata['error']}")
                continue
            
            # Add frames to metadata
            frame_metadata.extend(camera_metadata['frames'])
        
        logger.info(f"\nExtracted {len(frame_metadata)} total frames from {len(video_files)} cameras")
        
        # Initialize COLMAP status
        colmap_success = False
        
        # Run COLMAP if we have multiple views and not skipping
        if len(video_files) > 1 and not self.skip_colmap:
            # Sample images for COLMAP to speed up matching
            logger.info("\nPreparing images for COLMAP (sampling every 10th frame)...")
            colmap_images_dir = self.colmap_dir / "images"
            colmap_images_dir.mkdir(parents=True, exist_ok=True)
            sample_frames = frame_metadata[::10]
            copied = 0
            for frame_info in tqdm(sample_frames, desc="Preparing COLMAP images"):
                src = self.output_dir / frame_info['file_path']
                dst = colmap_images_dir / f"{frame_info['camera']}_{Path(src).name}"
                if src.exists():
                    shutil.copy2(src, dst)
                    copied += 1
            logger.info(f"  Copied {copied} images for COLMAP")

            logger.info("\nRunning COLMAP for multi-view calibration...")
            colmap_success = self.run_colmap_sfm(
                image_dir=str(colmap_images_dir),
                single_camera=False,
                gpu_index=0 if self.use_gpu else -1
            )
            
            if colmap_success:
                # Parse COLMAP output and generate transforms
                colmap_data = self.parse_colmap_output()
                transforms = self.generate_transforms_json(colmap_data, frame_metadata)
                
                # Save transforms.json
                transforms_path = self.output_dir / "transforms.json"
                with open(transforms_path, 'w') as f:
                    json.dump(transforms, f, indent=2)
                
                logger.info(f"✅ Saved transforms.json to {transforms_path}")
            else:
                logger.warning("COLMAP failed. You may need to provide manual calibration.")
        elif len(video_files) > 1 and self.skip_colmap:
            logger.info("Skipping COLMAP as requested (--skip_colmap). Will generate simple transforms.")
            colmap_success = False
        else:
            logger.info("Single video detected. Skipping COLMAP (not needed for single view)")
            colmap_success = False
        
        # Always generate transforms - either from COLMAP or with defaults
        if not colmap_success or len(video_files) == 1:
            # Generate simple transforms for single camera or COLMAP failure
            transforms = {
                'frames': frame_metadata,
                'camera_angle_x': 1.0,  # Default, will need calibration
                'fl_x': 1000.0,  # Default focal length
                'fl_y': 1000.0,
                'cx': 960,  # Assuming 1920 width
                'cy': 540,  # Assuming 1080 height
                'w': 1920,
                'h': 1080,
                'aabb_scale': 4,
                'scale': 1.0,
                'offset': [0.5, 0.5, 0.5]
            }
            
            transforms_path = self.output_dir / "transforms.json"
            with open(transforms_path, 'w') as f:
                json.dump(transforms, f, indent=2)
            
            logger.info(f"✅ Saved transforms.json to {transforms_path}")
        
        # Save processing metadata
        metadata = {
            'num_cameras': len(video_files),
            'camera_names': camera_names,
            'video_files': [str(vf) for vf in video_files],
            'total_frames': len(frame_metadata),
            'fps': fps,
            'skip_frames': skip_frames,
            'resize': resize,
            'colmap_success': colmap_success if len(video_files) > 1 else None
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Saved metadata to {metadata_path}")
        
        return {
            'success': True,
            'num_cameras': len(video_files),
            'total_frames': len(frame_metadata),
            'output_dir': str(self.output_dir)
        }
    
    def process_videos(self,
                      video_paths: List[str],
                      camera_names: Optional[List[str]] = None,
                      calibration_file: Optional[str] = None,
                      target_fps: float = 30.0,
                      extract_every_n: int = 1,
                      max_frames: int = -1) -> Dict:
        """
        Main processing pipeline for multi-view videos.
        
        Args:
            video_paths: List of video file paths
            camera_names: Optional camera names
            calibration_file: Optional pre-computed calibration
            target_fps: Target frame rate
            extract_every_n: Extract every N frames
            max_frames: Maximum frames to extract (-1 for all)
            
        Returns:
            Processing results
        """
        logger.info("="*60)
        logger.info("Starting 4DGS preprocessing pipeline")
        logger.info("="*60)
        
        # Step 1: Extract frames from videos
        logger.info("\n📹 Step 1: Extracting frames from videos...")
        
        if camera_names is None:
            camera_names = [f"cam{i}" for i in range(len(video_paths))]
        
        # Create camera configs
        cameras = []
        for i, (video_path, cam_name) in enumerate(zip(video_paths, camera_names)):
            cameras.append(CameraConfig(
                name=cam_name,
                video_path=video_path
            ))
        
        # Use VideoProcessor to extract frames
        processor = VideoProcessor(
            output_dir=str(self.output_dir),
            target_fps=target_fps,
            extract_every_n=extract_every_n,
            use_gpu=self.use_gpu
        )
        
        metadata = processor.process_multi_camera_videos(cameras)
        
        # Check if we have frames
        if not metadata['frames']:
            logger.error("No frames extracted!")
            return None
        
        logger.info(f"✅ Extracted {len(metadata['frames'])} frames total")
        
        # Step 2: Run COLMAP if no calibration provided
        colmap_data = None
        if calibration_file is None and not self.skip_colmap:
            logger.info("\n🎥 Step 2: Running camera calibration with COLMAP...")
            
            # Prepare images for COLMAP (combine all camera frames)
            colmap_images_dir = self.colmap_dir / "images"
            colmap_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy sample frames for COLMAP
            sample_frames = metadata['frames'][::10]  # Every 10th frame for speed
            for frame_info in tqdm(sample_frames, desc="Preparing COLMAP images"):
                src = self.output_dir / frame_info['file_path']
                dst = colmap_images_dir / f"{frame_info['camera']}_{Path(src).name}"
                if src.exists():
                    shutil.copy2(src, dst)
            
            # Run COLMAP
            success = self.run_colmap_sfm(str(colmap_images_dir))
            
            if success:
                colmap_data = self.parse_colmap_output()
                logger.info("✅ Camera calibration completed")
            else:
                logger.warning("⚠️  COLMAP failed, using default calibration")
        elif calibration_file is None and self.skip_colmap:
            logger.info("\n🎥 Step 2: Skipping COLMAP as requested (--skip_colmap)")
        else:
            logger.info(f"\n📂 Step 2: Loading calibration from {calibration_file}")
            with open(calibration_file, 'r') as f:
                colmap_data = json.load(f)
        
        # Step 3: Generate transforms.json
        logger.info("\n📝 Step 3: Generating transforms.json...")
        
        if colmap_data and colmap_data.get('images'):
            transforms = self.generate_transforms_json(colmap_data, metadata['frames'])
        else:
            # Fallback: generate simple transforms
            transforms = self.generate_simple_transforms(metadata)
        
        # Save transforms
        transforms_path = self.output_dir / "transforms.json"
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        logger.info(f"✅ Saved transforms to {transforms_path}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("🎉 Preprocessing completed successfully!")
        logger.info(f"📂 Output directory: {self.output_dir}")
        logger.info(f"📊 Statistics:")
        logger.info(f"   - Total frames: {len(metadata['frames'])}")
        logger.info(f"   - Cameras: {len(metadata['cameras'])}")
        logger.info(f"   - Resolution: {transforms.get('w', 0)}x{transforms.get('h', 0)}")
        logger.info("="*60)
        
        return {
            'output_dir': str(self.output_dir),
            'transforms_path': str(transforms_path),
            'metadata': metadata,
            'colmap_data': colmap_data
        }
    
    def generate_simple_transforms(self, metadata: Dict) -> Dict:
        """
        Generate simple transforms without COLMAP.
        Uses identity poses and estimates intrinsics.
        
        Args:
            metadata: Video processing metadata
            
        Returns:
            Transforms dictionary
        """
        # Get first camera's resolution
        first_cam = list(metadata['cameras'].values())[0]
        w, h = first_cam['resolution']
        
        # Estimate focal length (simple approximation)
        focal = max(w, h)
        
        transforms = {
            'frames': [],
            'camera_angle_x': 2 * np.arctan(w / (2 * focal)),
            'fl_x': focal,
            'fl_y': focal,
            'cx': w / 2,
            'cy': h / 2,
            'w': w,
            'h': h,
            'aabb_scale': 4,
            'scale': 1.0,
            'offset': [0.5, 0.5, 0.5]
        }
        
        # Create identity poses for each frame
        for frame_info in metadata['frames']:
            # Simple circular camera arrangement
            cam_idx = list(metadata['cameras'].keys()).index(frame_info['camera'])
            angle = 2 * np.pi * cam_idx / len(metadata['cameras'])
            
            # Camera position on circle
            radius = 3.0
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0.0
            
            # Look-at matrix (camera looking at origin)
            c2w = look_at_matrix([x, y, z], [0, 0, 0], [0, 1, 0])
            
            frame_data = {
                'file_path': frame_info['file_path'],
                'transform_matrix': c2w.tolist(),
                'time': frame_info['time'],
                'camera': frame_info['camera']
            }
            
            transforms['frames'].append(frame_data)
        
        return transforms


def quaternion_to_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx**2 + qy**2)
    return R


def look_at_matrix(eye, center, up):
    """Create look-at matrix."""
    eye = np.array(eye)
    center = np.array(center)
    up = np.array(up)
    
    f = center - eye
    f = f / np.linalg.norm(f)
    
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    
    u = np.cross(s, f)
    
    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = eye
    
    return m


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess multi-view videos for 4D Gaussian Splatting training',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Input options - either folder or individual videos
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video_folder', '-f', type=str,
                           help='Path to folder containing video files\n(Uses video filenames as camera names)')
    input_group.add_argument('--videos', '-v', nargs='+', type=str, 
                           help='Individual video file paths')
    
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--camera_names', nargs='+', default=None,
                       help='Camera names for --videos option\n(For --video_folder, filenames are used automatically)')
    parser.add_argument('--calibration', type=str, default=None,
                       help='Pre-computed calibration file')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Target frame rate (default: 30)')
    parser.add_argument('--skip_frames', type=int, default=1,
                       help='Extract every N frames (default: 1)')
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='Resize frames to WIDTH HEIGHT (e.g., 1920 1080)')
    parser.add_argument('--max_frames', type=int, default=-1,
                       help='Maximum frames to extract (-1 for all)')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU acceleration for COLMAP')
    parser.add_argument('--skip_colmap', action='store_true',
                       help='Skip COLMAP and generate simple transforms (identity poses, estimated intrinsics)')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = MultiViewPreprocessor(
        output_dir=args.output,
        use_gpu=args.use_gpu,
        skip_colmap=args.skip_colmap
    )
    
    # Handle folder input
    if args.video_folder:
        logger.info(f"Processing all videos in folder: {args.video_folder}")
        
        resize_tuple = tuple(args.resize) if args.resize else None
        result = preprocessor.process_video_folder(
            video_folder=args.video_folder,
            output_dir=args.output,
            fps=int(args.fps),
            skip_frames=args.skip_frames,
            resize=resize_tuple,
            start_time=0.0,
            end_time=None
        )
        
        if result['success']:
            print("\n" + "="*60)
            print("✅ PREPROCESSING COMPLETE")
            print("="*60)
            print(f"Cameras processed: {result['num_cameras']}")
            print(f"Total frames: {result['total_frames']}")
            print(f"Output directory: {result['output_dir']}")
            print("\nNext step:")
            print(f"  python tools/train.py --data_root {result['output_dir']} --out_dir model/")
        else:
            logger.error("Preprocessing failed")
            sys.exit(1)
    
    # Handle individual video files
    else:
        # Validate inputs
        for video in args.videos:
            if not os.path.exists(video):
                logger.error(f"Video file not found: {video}")
                sys.exit(1)
        
        # Use filenames as camera names if not provided
        if args.camera_names is None:
            args.camera_names = [Path(v).stem for v in args.videos]
            logger.info(f"Using video filenames as camera names: {args.camera_names}")
        elif len(args.camera_names) != len(args.videos):
            logger.error(f"Number of camera names ({len(args.camera_names)}) must match videos ({len(args.videos)})")
            sys.exit(1)
        
        # Process individual videos
        result = preprocessor.process_videos(
            video_paths=args.videos,
            camera_names=args.camera_names,
            calibration_file=args.calibration,
            target_fps=args.fps,
            extract_every_n=args.skip_frames,
            max_frames=args.max_frames
        )
        
        if result:
            print("\n" + "="*60)
            print("✅ PREPROCESSING COMPLETE")
            print("="*60)
            print(f"Output directory: {result['output_dir']}")
            print("\nNext step:")
            print(f"  python tools/train.py --data_root {result['output_dir']} --out_dir model/")


if __name__ == '__main__':
    main()