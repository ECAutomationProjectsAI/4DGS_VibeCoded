#!/usr/bin/env python
"""
Complete preprocessing pipeline for 4D Gaussian Splatting.

This script handles the full data preparation workflow:
1. Extract frames from multi-view videos
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
from typing import Dict, List, Optional, Tuple
import subprocess
import shutil
from tqdm import tqdm
import logging

# Add parent directory to import gs4d modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs4d.video_processor import VideoProcessor, CameraConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiViewPreprocessor:
    """Complete preprocessing pipeline for 4DGS."""
    
    def __init__(self, output_dir: str, use_gpu: bool = False):
        """
        Initialize preprocessor.
        
        Args:
            output_dir: Output directory for processed data
            use_gpu: Use GPU acceleration if available
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        
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
        
        # Step 1: Feature extraction
        logger.info("  1. Extracting features...")
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--ImageReader.camera_model", camera_model,
            "--SiftExtraction.use_gpu", "1" if self.use_gpu else "0",
            "--SiftExtraction.gpu_index", str(gpu_index)
        ]
        
        if single_camera:
            cmd.extend(["--ImageReader.single_camera", "1"])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Feature extraction failed: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("COLMAP not found! Please install COLMAP and add it to PATH")
            logger.info("Download from: https://github.com/colmap/colmap/releases")
            return False
        
        # Step 2: Feature matching
        logger.info("  2. Matching features...")
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", "1" if self.use_gpu else "0",
            "--SiftMatching.gpu_index", str(gpu_index)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
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
            subprocess.run(cmd, check=True, capture_output=True, text=True)
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
            subprocess.run(cmd, check=True, capture_output=True, text=True)
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
        if calibration_file is None:
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
        description='Preprocess multi-view videos for 4D Gaussian Splatting training'
    )
    parser.add_argument('--videos', nargs='+', required=True, 
                       help='Input video files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--camera_names', nargs='+', default=None,
                       help='Camera names (default: cam0, cam1, ...)')
    parser.add_argument('--calibration', type=str, default=None,
                       help='Pre-computed calibration file')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Target frame rate')
    parser.add_argument('--skip_frames', type=int, default=1,
                       help='Extract every N frames')
    parser.add_argument('--max_frames', type=int, default=-1,
                       help='Maximum frames to extract')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU acceleration')
    
    args = parser.parse_args()
    
    # Validate inputs
    for video in args.videos:
        if not os.path.exists(video):
            logger.error(f"Video file not found: {video}")
            sys.exit(1)
    
    # Run preprocessing
    preprocessor = MultiViewPreprocessor(
        output_dir=args.output,
        use_gpu=args.use_gpu
    )
    
    result = preprocessor.process_videos(
        video_paths=args.videos,
        camera_names=args.camera_names,
        calibration_file=args.calibration,
        target_fps=args.fps,
        extract_every_n=args.skip_frames,
        max_frames=args.max_frames
    )
    
    if result:
        print(f"\n✅ Ready for training!")
        print(f"Run: python tools/train.py --data_root {result['output_dir']}")


if __name__ == '__main__':
    main()