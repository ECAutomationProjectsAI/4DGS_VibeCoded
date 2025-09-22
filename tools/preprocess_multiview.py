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
    
    def __init__(self,
                 output_dir: str,
                 use_gpu: bool = False,
                 skip_colmap: bool = False,
                 colmap_threads: int = 0,
                 colmap_max_image_size: int = 0,
                 colmap_max_num_features: int = 0,
                 colmap_matcher: str = "auto",
                 colmap_sample_rate: int = 10,
                 colmap_mapped_groups: int = 3):
        """
        Initialize preprocessor.
        
        Args:
            output_dir: Output directory for processed data
            use_gpu: Use GPU acceleration if available (for video processing, not COLMAP)
            skip_colmap: Skip running COLMAP even for multi-view
            colmap_threads: Limit SIFT extraction threads (0 = default)
            colmap_max_image_size: Downscale images for SIFT (0 = original)
            colmap_max_num_features: Limit max SIFT features per image (0 = default)
            colmap_matcher: 'auto' | 'exhaustive' | 'sequential'
            colmap_sample_rate: Use every Nth frame for COLMAP to reduce memory/work
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        self.skip_colmap = skip_colmap
        self.colmap_threads = int(colmap_threads) if colmap_threads is not None else 0
        self.colmap_max_image_size = int(colmap_max_image_size) if colmap_max_image_size is not None else 0
        self.colmap_max_num_features = int(colmap_max_num_features) if colmap_max_num_features is not None else 0
        self.colmap_matcher = colmap_matcher
        self.colmap_sample_rate = max(1, int(colmap_sample_rate))
        self.colmap_mapped_groups = max(1, int(colmap_mapped_groups))
        
        # Subdirectories
        self.frames_dir = self.output_dir / "frames"
        self.colmap_dir = self.output_dir / "colmap"
        self.masks_dir = self.output_dir / "masks"
        
        # Logs
        self.logs_dir = self.colmap_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def run_colmap_sfm(self, 
                       image_dir: str,
                       camera_model: str = "OPENCV",
                       single_camera: bool = False,
                       gpu_index: int = 0) -> bool:
        """
        Run COLMAP Structure-from-Motion pipeline with streaming logs and memory controls.
        """
        logger.info("Running COLMAP SfM pipeline...")
        
        # Create COLMAP workspace
        self.colmap_dir.mkdir(exist_ok=True)
        database_path = self.colmap_dir / "database.db"
        
        # Prepare a clean environment for headless COLMAP
        colmap_env = os.environ.copy()
        colmap_env.pop('QT_PLUGIN_PATH', None)
        colmap_env.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
        colmap_env['QT_QPA_PLATFORM'] = 'offscreen'
        # Avoid over-threading (can cause RAM spikes)
        colmap_env.setdefault('OMP_NUM_THREADS', str(self.colmap_threads or os.cpu_count() or 4))

        # Util: run command and stream to logger and file
        def _run_and_stream(cmd: List[str], log_name: str) -> int:
            log_path = self.logs_dir / log_name
            logger.info(f"[COLMAP] {' '.join(cmd)}")
            with open(log_path, 'w', encoding='utf-8') as lf:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=colmap_env)
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        logger.info(line)
                        lf.write(line + "\n")
                return proc.wait()

        # Helper: check if a COLMAP subcommand supports an option
        def _colmap_has_option(subcmd: str, option: str) -> bool:
            try:
                help_proc = subprocess.run(["colmap", subcmd, "-h"], check=False, capture_output=True, text=True, env=colmap_env)
                return option in (help_proc.stdout or "")
            except Exception:
                return False

        # Step 1: Feature extraction
        logger.info("  1) Extracting features...")
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--ImageReader.camera_model", camera_model
        ]
        # SIFT options (favor stability and lower memory)
        if _colmap_has_option("feature_extractor", "--SiftExtraction.use_gpu"):
            cmd.extend(["--SiftExtraction.use_gpu", "0"])  # safer in headless containers
        if self.colmap_threads > 0 and _colmap_has_option("feature_extractor", "--SiftExtraction.num_threads"):
            cmd.extend(["--SiftExtraction.num_threads", str(self.colmap_threads)])
        if self.colmap_max_image_size > 0 and _colmap_has_option("feature_extractor", "--SiftExtraction.max_image_size"):
            cmd.extend(["--SiftExtraction.max_image_size", str(self.colmap_max_image_size)])
        if self.colmap_max_num_features > 0 and _colmap_has_option("feature_extractor", "--SiftExtraction.max_num_features"):
            cmd.extend(["--SiftExtraction.max_num_features", str(self.colmap_max_num_features)])
        if single_camera:
            cmd.extend(["--ImageReader.single_camera", "1"])
        ret = _run_and_stream(cmd, "01_feature_extractor.log")
        if ret != 0:
            logger.error(f"Feature extraction failed with code {ret}")
            return False

        # Step 2: Feature matching
        # Decide matcher
        img_exts = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
        num_images = len([f for f in os.listdir(image_dir) if f.endswith(img_exts)])
        matcher = self.colmap_matcher
        if matcher == "auto":
            matcher = "sequential" if num_images > 800 else "exhaustive"
        logger.info(f"  2) Matching features ({matcher}, images={num_images})...")
        if matcher == "sequential":
            cmd = [
                "colmap", "sequential_matcher",
                "--database_path", str(database_path),
                "--SequentialMatching.overlap", "5"
            ]
        else:
            cmd = ["colmap", "exhaustive_matcher", "--database_path", str(database_path)]
        if _colmap_has_option(cmd[1], "--SiftMatching.use_gpu"):
            cmd.extend(["--SiftMatching.use_gpu", "0"])  # safer
        ret = _run_and_stream(cmd, "02_matcher.log")
        if ret != 0:
            logger.error(f"Feature matching failed with code {ret}")
            return False

        # Step 3: Sparse reconstruction (mapper)
        logger.info("  3) Running sparse reconstruction (mapper)...")
        sparse_dir = self.colmap_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        cmd = [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir)
        ]
        ret = _run_and_stream(cmd, "03_mapper.log")
        if ret != 0:
            logger.error(f"Sparse reconstruction failed with code {ret}")
            return False

        # Step 4: Convert to text format
        logger.info("  4) Converting to text format...")
        sparse_txt_dir = self.colmap_dir / "sparse_txt"
        sparse_txt_dir.mkdir(exist_ok=True)
        recon_dirs = list(sparse_dir.glob("*"))
        if not recon_dirs:
            logger.error("No reconstruction found!")
            return False
        recon_dir = recon_dirs[0]
        cmd = [
            "colmap", "model_converter",
            "--input_path", str(recon_dir),
            "--output_path", str(sparse_txt_dir),
            "--output_type", "TXT"
        ]
        ret = _run_and_stream(cmd, "04_model_converter.log")
        if ret != 0:
            logger.error(f"Model conversion failed with code {ret}")
            return False
        logger.info("✅ COLMAP SfM completed successfully!")
        return True
    
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
                                frame_metadata: List[Dict],
                                name_map: Optional[Dict[str, str]] = None) -> Dict:
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
        
        # Build quick lookup for metadata by original file path
        meta_by_path = {fr['file_path']: fr for fr in frame_metadata}
        
        # Build a per-camera pose map from COLMAP images (assumes static cameras)
        cam_pose_map = {}
        for image_name, img_data in colmap_data.get('images', {}).items():
            # Infer camera name from image_name: expect it was copied as "{camera}_{basename}"
            inferred_cam = None
            if '_' in image_name:
                inferred_cam = image_name.split('_', 1)[0]
            else:
                # Try alternative delimiters if needed
                for delim in ['-', ' ']:
                    if delim in image_name:
                        inferred_cam = image_name.split(delim, 1)[0]
                        break
            if inferred_cam is None:
                continue
            # Convert quaternion and translation to c2w
            qw, qx, qy, qz = img_data['quaternion']
            tx, ty, tz = map(float, img_data['translation'])
            R = quaternion_to_matrix(qw, qx, qy, qz)
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ np.array([tx, ty, tz])
            # Store one pose per camera (first occurrence wins)
            cam_pose_map.setdefault(inferred_cam, c2w)

        # Now generate a transform entry for every extracted frame, using the camera's pose
        for fr in frame_metadata:
            cam = fr.get('camera', 'cam0')
            if cam not in cam_pose_map:
                # Skip frames for cameras not calibrated by COLMAP
                # (fallback will include them later or user can rerun COLMAP)
                continue
            c2w = cam_pose_map[cam]
            frame_data = {
                'file_path': fr['file_path'],
                'transform_matrix': c2w.tolist(),
                'time': fr.get('time', 0.0),
                'camera': cam
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
                           start_frame: int = 0,
                           end_frame: Optional[int] = None) -> Dict:
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
                start_frame=start_frame,
                end_frame=end_frame
            )
            
            # Check for errors
            if 'error' in camera_metadata:
                logger.error(f"Failed to process {video_file.name}: {camera_metadata['error']}")
                continue
            
            # Add frames to metadata
            frame_metadata.extend(camera_metadata['frames'])
        
        logger.info(f"\nExtracted {len(frame_metadata)} total frames from {len(video_files)} cameras")
        
        # Move frames/ -> extracted_frames/
        extracted_frames_dir = self.output_dir / "extracted_frames"
        extracted_frames_dir.mkdir(parents=True, exist_ok=True)
        orig_frames_dir = self.output_dir / "frames"
        if orig_frames_dir.exists():
            for cam_dir in orig_frames_dir.glob("*"):
                if cam_dir.is_dir():
                    dst_cam_dir = extracted_frames_dir / cam_dir.name
                    dst_cam_dir.mkdir(parents=True, exist_ok=True)
                    for img_path in cam_dir.glob("*"):
                        shutil.move(str(img_path), str(dst_cam_dir / img_path.name))
            # remove empty original dir structure
            try:
                for cam_dir in orig_frames_dir.glob("*"):
                    if cam_dir.is_dir():
                        cam_dir.rmdir()
                orig_frames_dir.rmdir()
            except Exception:
                pass
        
        # Update file paths in metadata to point to extracted_frames/
        for fr in frame_metadata:
            fr['file_path'] = fr['file_path'].replace('frames/', 'extracted_frames/')
        
        # Create frames_mapped/ with per-frame folders containing one image per camera
        frames_mapped_dir = self.output_dir / "frames_mapped"
        frames_mapped_dir.mkdir(parents=True, exist_ok=True)
        
        # Build mapping by absolute frame index
        frames_by_idx = {}
        cams_set = set(camera_names)
        for fr in frame_metadata:
            idx = int(fr.get('frame_idx', -1))
            if idx < 0:
                # derive from filename if missing
                try:
                    idx = int(Path(fr['file_path']).stem.split('_')[-1])
                except Exception:
                    continue
            cam = fr.get('camera', 'cam0')
            frames_by_idx.setdefault(idx, {})[cam] = fr['file_path']
        
        # Create ordered list of indices present in all cameras
        ordered_indices = sorted([idx for idx, cammap in frames_by_idx.items() if cams_set.issubset(set(cammap.keys()))])
        logger.info(f"Mapped {len(ordered_indices)} frame groups across {len(cams_set)} cameras")
        
        # Materialize mapping folders
        for i, idx in enumerate(ordered_indices, start=1):
            group_dir = frames_mapped_dir / f"frame{i:06d}"
            group_dir.mkdir(parents=True, exist_ok=True)
            for cam in sorted(cams_set):
                src_rel = frames_by_idx[idx][cam]
                src_abs = self.output_dir / src_rel
                dst_abs = group_dir / f"{cam}.jpg"
                if src_abs.exists():
                    shutil.copy2(src_abs, dst_abs)
        
        # Initialize COLMAP status
        colmap_success = False
        
        # Run COLMAP on first N mapped frames if multi-view and not skipping
        if len(video_files) > 1 and not self.skip_colmap:
            colmap_images_dir = self.colmap_dir / "images"
            colmap_images_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            name_map = {}
            # Build ordered group list with original idx mapping
            # Reconstruct using ordered_indices and frames_by_idx from above
            # Enumerate in the same order we created frames_mapped
            for gi, idx in enumerate(ordered_indices, start=1):
                if gi > self.colmap_mapped_groups:
                    break
                cammap = frames_by_idx[idx]
                group_name = f"frame{gi:06d}"
                for cam, rel_path in cammap.items():
                    src_abs = self.output_dir / rel_path
                    colmap_name = f"{group_name}_{cam}.jpg"
                    dst_abs = colmap_images_dir / colmap_name
                    if src_abs.exists():
                        shutil.copy2(src_abs, dst_abs)
                        name_map[colmap_name] = rel_path
                        copied += 1
            # Optionally write name_map for debugging
            with open(self.colmap_dir / "name_map.json", 'w') as f:
                json.dump(name_map, f, indent=2)
            logger.info(f"Prepared {copied} images for COLMAP from first {self.colmap_mapped_groups} mapped frames")

            logger.info("\nRunning COLMAP for multi-view calibration...")
            colmap_success = self.run_colmap_sfm(
                image_dir=str(colmap_images_dir),
                single_camera=False,
                gpu_index=0 if self.use_gpu else -1
            )
            
            if colmap_success:
                # Parse COLMAP output and generate transforms (use name_map to match files)
                colmap_data = self.parse_colmap_output()
                # Dump COLMAP parsed data for debugging
                with open(self.colmap_dir / "colmap_parsed.json", 'w') as f:
                    json.dump(colmap_data, f, indent=2)
                transforms = self.generate_transforms_json(colmap_data, frame_metadata, name_map=name_map)
                
                # Validate calibrated frames
                n_calibrated = len(transforms.get('frames', []))
                logger.info(f"COLMAP produced {n_calibrated} calibrated frames")
                if n_calibrated == 0:
                    logger.error("❌ No calibrated frames found after COLMAP! Aborting.")
                    logger.error("Troubleshooting tips:")
                    logger.error("  - Verify extracted_frames/ contains images for all cameras in the selected range")
                    logger.error("  - Verify frames_mapped/ has at least one frame group with all cameras")
                    logger.error("  - Increase --colmap_mapped_groups (e.g., 5 or 10)")
                    logger.error("  - Ensure start/end frame range is correct and not empty")
                    logger.error("  - Check colmap/logs/*.log for errors")
                    raise RuntimeError("COLMAP returned zero calibrated frames")
                
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
            
            n_fallback = len(transforms.get('frames', []))
            logger.info(f"✅ Saved fallback transforms.json with {n_fallback} frames to {transforms_path}")
        
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
                      max_frames: int = -1,
                      start_frame: int = 0,
                      end_frame: Optional[int] = None) -> Dict:
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
        
        # Apply strict frame range to camera configs
        for cam in cameras:
            cam.start_frame = start_frame
            cam.end_frame = end_frame
        metadata = processor.process_multi_camera_videos(cameras, sync_method="frame")
        
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
            cam_idx = list(metadata['cameras'].keys()).index(frame_info['camera']) if metadata['cameras'] else 0
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
        description='Preprocess multi-view videos (folder-only) for 4D Gaussian Splatting training',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Folder-only input
    parser.add_argument('input_dir', type=str, help='Path to folder containing video files (camera names derived from filenames)')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory')
    parser.add_argument('--calibration', type=str, default=None, help='Pre-computed calibration file (optional)')

    # Strict frame index selection
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame index (inclusive)')
    parser.add_argument('--end_frame', type=int, default=None, help='End frame index (exclusive)')

    # Core options
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help='Resize frames to WIDTH HEIGHT (e.g., 1920 1080)')
    parser.add_argument('--extract-every', type=int, default=1, help='Extract every Nth frame (default: 1 = every frame)')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU acceleration for video preprocessing (not COLMAP)')

    # COLMAP controls (kept minimal)
    parser.add_argument('--skip_colmap', action='store_true', help='Skip running COLMAP')
    parser.add_argument('--colmap_mapped_groups', type=int, default=3, help='Number of mapped frame groups to use for COLMAP subset (default: 3)')

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = MultiViewPreprocessor(
        output_dir=args.output,
        use_gpu=args.use_gpu,
        skip_colmap=args.skip_colmap,
        colmap_mapped_groups=args.colmap_mapped_groups
    )

    # Validate input
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input must be a directory of videos: {args.input_dir}")
        sys.exit(1)

    logger.info(f"Processing all videos in folder: {args.input_dir}")

    resize_tuple = tuple(args.resize) if args.resize else None
    result = preprocessor.process_video_folder(
        video_folder=args.input_dir,
        output_dir=args.output,
        fps=30,
        skip_frames=args.extract_every,
        resize=resize_tuple,
        start_frame=args.start_frame,
        end_frame=args.end_frame
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


if __name__ == '__main__':
    print("This script has been merged into tools/preprocess.py. Please use:\n")
    print("  python3 tools/preprocess.py /path/to/videos -o /path/to/dataset [--resize W H] [--extract-every N]\n")
    sys.exit(1)
