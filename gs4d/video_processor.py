"""Video processing module for 4D Gaussian Splatting.

Handles:
- Single or multi-camera video sequences
- Automatic frame extraction and synchronization
- Camera calibration data integration
- Various video formats (mp4, avi, mov, etc.)
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
from dataclasses import dataclass
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration for a video sequence."""
    name: str
    video_path: str
    calibration: Optional[Dict[str, Any]] = None
    fps: float = 30.0
    start_frame: int = 0
    end_frame: Optional[int] = None
    sync_offset: float = 0.0  # Time offset in seconds for synchronization


class VideoProcessor:
    """Process video sequences for 4DGS training."""
    
    def __init__(
        self,
        output_dir: str,
        target_fps: float = 30.0,
        resize: Optional[Tuple[int, int]] = None,
        extract_every_n: int = 1,
        use_gpu: bool = False
    ):
        """
        Initialize video processor.
        
        Args:
            output_dir: Directory to save extracted frames
            target_fps: Target FPS for extraction
            resize: Optional (width, height) to resize frames
            extract_every_n: Extract every N frames (for faster processing)
            use_gpu: Use GPU acceleration if available (requires opencv-contrib-python)
        """
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.resize = resize
        self.extract_every_n = extract_every_n
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
    def process_single_camera_video(
        self,
        video_path: str,
        camera_name: str = "cam0",
        calibration: Optional[Dict[str, Any]] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a single camera video.
        
        Args:
            video_path: Path to video file
            camera_name: Name for this camera
            calibration: Optional camera calibration data
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Metadata dictionary with frame paths and camera info
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Create output directory for this camera
        cam_dir = self.output_dir / "frames" / camera_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            error_msg = f"Failed to open video: {video_path} - File may be corrupted or unsupported format"
            logger.error(error_msg)
            # Return minimal metadata for failed video
            return {
                "camera_name": camera_name,
                "error": error_msg,
                "calibration": calibration,
                "frames": [],
                "fps": 0,
                "resolution": [0, 0]
            }
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame range (frame indices, inclusive-exclusive)
        start_frame = max(0, int(start_frame))
        end_frame = int(end_frame) if end_frame is not None else total_frames
        end_frame = min(end_frame, total_frames)
        if end_frame <= start_frame:
            cap.release()
            return {
                "camera_name": camera_name,
                "error": f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}",
                "calibration": calibration,
                "frames": [],
                "fps": fps,
                "resolution": [width, height]
            }
        
        # Set up calibration if not provided
        if calibration is None:
            # Use default calibration (simple pinhole model)
            focal = max(width, height)
            calibration = {
                "fx": focal,
                "fy": focal,
                "cx": width / 2,
                "cy": height / 2,
                "k1": 0, "k2": 0, "p1": 0, "p2": 0,
                "width": width,
                "height": height
            }
        
        # Extract frames
        frames_metadata = []
        frame_idx = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        total_to_extract = max(0, (end_frame - start_frame + (self.extract_every_n - 1)) // self.extract_every_n)
        pbar = tqdm(total=total_to_extract, desc=f"Extracting {camera_name}")
        
        while cap.isOpened() and frame_idx < end_frame - start_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.extract_every_n == 0:
                # Resize if needed
                if self.resize:
                    frame = cv2.resize(frame, self.resize)
                
                # Save frame
                frame_filename = f"{camera_name}_{frame_idx:06d}.jpg"
                frame_path = cam_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                # Add to metadata (strictly by frame index)
                abs_frame_idx = start_frame + frame_idx
                frames_metadata.append({
                    "file_path": f"frames/{camera_name}/{frame_filename}",
                    "time": abs_frame_idx / fps,
                    "camera": camera_name,
                    "frame_idx": abs_frame_idx
                })
                
                pbar.update(1)
            
            frame_idx += 1
        
        cap.release()
        pbar.close()
        
        return {
            "camera_name": camera_name,
            "calibration": calibration,
            "frames": frames_metadata,
            "fps": fps,
            "resolution": [width, height] if not self.resize else list(self.resize)
        }
    
    def process_multi_camera_videos(
        self,
        cameras: List[CameraConfig],
        sync_method: str = "frame"
    ) -> Dict[str, Any]:
        """
        Process multiple synchronized camera videos.
        
        Args:
            cameras: List of camera configurations
            sync_method: Synchronization method ("timestamp", "frame", "manual")
            
        Returns:
            Combined metadata for all cameras
        """
        all_metadata = {
            "cameras": {},
            "frames": [],
            "sync_method": sync_method,
            "failed_videos": []  # Track failed videos
        }
        
        successful_cameras = 0
        
        # Process each camera
        for cam_config in cameras:
            logger.info(f"Processing camera: {cam_config.name}")
            
            # Use explicit frame indices strictly
            try:
                cam_metadata = self.process_single_camera_video(
                    video_path=cam_config.video_path,
                    camera_name=cam_config.name,
                    calibration=cam_config.calibration,
                    start_frame=cam_config.start_frame,
                    end_frame=cam_config.end_frame
                )
                
                # Check if video processing failed
                if "error" in cam_metadata:
                    all_metadata["failed_videos"].append({
                        "camera": cam_config.name,
                        "path": cam_config.video_path,
                        "error": cam_metadata["error"]
                    })
                    logger.warning(f"Skipping failed video: {cam_config.name}")
                    continue
                
                # Only add successful cameras
                if cam_metadata["frames"]:
                    successful_cameras += 1
                    all_metadata["cameras"][cam_config.name] = {
                        "calibration": cam_metadata["calibration"],
                        "fps": cam_metadata["fps"],
                        "resolution": cam_metadata["resolution"]
                    }
                    
                    # Add frames with camera info
                    for frame in cam_metadata["frames"]:
                        all_metadata["frames"].append(frame)
                        
            except Exception as e:
                error_msg = f"Unexpected error processing {cam_config.name}: {str(e)}"
                logger.error(error_msg)
                all_metadata["failed_videos"].append({
                    "camera": cam_config.name,
                    "path": cam_config.video_path,
                    "error": error_msg
                })
        
        # Check if we have at least one successful camera
        if successful_cameras == 0:
            logger.error("No videos could be processed successfully!")
            logger.error("Failed videos:")
            for failed in all_metadata["failed_videos"]:
                logger.error(f"  - {failed['camera']}: {failed['error']}")
            raise RuntimeError("All videos failed to process. Please check your input files.")
        
        # Sort frames by timestamp for proper temporal ordering
        if all_metadata["frames"]:
            all_metadata["frames"].sort(key=lambda x: (x["time"], x["camera"]))
        
        # Print summary
        logger.info("="*50)
        logger.info(f"Processing Summary:")
        logger.info(f"  Successful cameras: {successful_cameras}/{len(cameras)}")
        logger.info(f"  Total frames extracted: {len(all_metadata['frames'])}")
        
        if all_metadata["failed_videos"]:
            logger.warning(f"  Failed videos: {len(all_metadata['failed_videos'])}")
            for failed in all_metadata["failed_videos"]:
                logger.warning(f"    - {failed['camera']}: {failed['path']}")
                logger.warning(f"      Error: {failed['error']}")
        logger.info("="*50)
        
        # Save metadata
        self._save_metadata(all_metadata)
        
        return all_metadata
    
    def process_video_directory(
        self,
        video_dir: str,
        calibration_file: Optional[str] = None,
        video_extension: str = "mp4"
    ) -> Dict[str, Any]:
        """
        Process all videos in a directory as multi-camera setup.
        
        Args:
            video_dir: Directory containing video files
            calibration_file: Optional JSON file with calibrations
            video_extension: Video file extension to look for
            
        Returns:
            Combined metadata
        """
        video_dir = Path(video_dir)
        video_files = sorted(video_dir.glob(f"*.{video_extension}"))
        
        if not video_files:
            raise FileNotFoundError(f"No {video_extension} files found in {video_dir}")
        
        # Load calibrations if provided
        calibrations = {}
        if calibration_file and Path(calibration_file).exists():
            with open(calibration_file, 'r') as f:
                calibrations = json.load(f)
        
        # Create camera configs
        cameras = []
        for idx, video_file in enumerate(video_files):
            cam_name = f"cam{idx}"
            
            # Try to extract camera name from filename
            if "_" in video_file.stem:
                potential_cam_name = video_file.stem.split("_")[0]
                if potential_cam_name.startswith("cam"):
                    cam_name = potential_cam_name
            
            cameras.append(CameraConfig(
                name=cam_name,
                video_path=str(video_file),
                calibration=calibrations.get(cam_name)
            ))
        
        return self.process_multi_camera_videos(cameras)
    
    def extract_colmap_format(
        self,
        video_path: str,
        colmap_dir: str,
        camera_model: str = "PINHOLE"
    ) -> None:
        """
        Extract frames in COLMAP format for SfM reconstruction.
        
        Args:
            video_path: Path to video
            colmap_dir: Output directory for COLMAP
            camera_model: Camera model type
        """
        colmap_dir = Path(colmap_dir)
        images_dir = colmap_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames
        metadata = self.process_single_camera_video(
            video_path,
            camera_name="colmap",
            start_time=0
        )
        
        # Move frames to COLMAP structure
        for frame_info in metadata["frames"]:
            src = self.output_dir / frame_info["file_path"]
            dst = images_dir / f"frame_{frame_info['frame_idx']:06d}.jpg"
            if src.exists():
                shutil.copy2(src, dst)
        
        # Create cameras.txt
        calib = metadata["calibration"]
        with open(colmap_dir / "cameras.txt", 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 {camera_model} {calib['width']} {calib['height']} ")
            f.write(f"{calib['fx']} {calib['fy']} {calib['cx']} {calib['cy']}\n")
        
        logger.info(f"Extracted {len(metadata['frames'])} frames for COLMAP")
    
    def create_transforms_json(
        self,
        metadata: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create transforms.json compatible with our dataloader.
        
        Args:
            metadata: Metadata from video processing
            output_path: Where to save the JSON
            
        Returns:
            Transforms dictionary
        """
        if output_path is None:
            output_path = self.output_dir / "transforms.json"
        
        transforms = {
            "fl_x": 0,  # Will be set per camera
            "fl_y": 0,
            "cx": 0,
            "cy": 0,
            "frames": []
        }
        
        # Group frames by timestamp
        frames_by_time = {}
        for frame in metadata["frames"]:
            time = frame["time"]
            if time not in frames_by_time:
                frames_by_time[time] = []
            frames_by_time[time].append(frame)
        
        # Create transform entries
        for time, frames in sorted(frames_by_time.items()):
            for frame_info in frames:
                cam_name = frame_info["camera"]
                cam_data = metadata["cameras"][cam_name]
                calib = cam_data["calibration"]
                
                # Simple identity transform (assuming cameras are pre-calibrated)
                # In practice, you'd compute this from COLMAP or other SfM
                transform_matrix = np.eye(4).tolist()
                
                transforms["frames"].append({
                    "file_path": frame_info["file_path"],
                    "time": time,
                    "transform_matrix": transform_matrix,
                    "camera": cam_name,
                    "fl_x": calib["fx"],
                    "fl_y": calib["fy"],
                    "cx": calib["cx"],
                    "cy": calib["cy"],
                    "w": cam_data["resolution"][0],
                    "h": cam_data["resolution"][1]
                })
        
        # Set default camera params (from first camera)
        if transforms["frames"]:
            first = transforms["frames"][0]
            transforms["fl_x"] = first["fl_x"]
            transforms["fl_y"] = first["fl_y"]
            transforms["cx"] = first["cx"]
            transforms["cy"] = first["cy"]
            transforms["w"] = first["w"]
            transforms["h"] = first["h"]
        
        # Save
        with open(output_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        return transforms
    
    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to JSON."""
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")


class VideoDataset:
    """Dataset wrapper for video sequences."""
    
    def __init__(self, root_dir: str):
        """
        Initialize video dataset.
        
        Args:
            root_dir: Root directory with extracted frames and metadata
        """
        self.root_dir = Path(root_dir)
        self.metadata_path = self.root_dir / "metadata.json"
        self.transforms_path = self.root_dir / "transforms.json"
        
        # Load metadata
        if self.transforms_path.exists():
            with open(self.transforms_path, 'r') as f:
                self.transforms = json.load(f)
        elif self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            # Create transforms from metadata
            processor = VideoProcessor(str(self.root_dir))
            self.transforms = processor.create_transforms_json(metadata)
        else:
            raise FileNotFoundError(f"No metadata found in {root_dir}")
    
    def get_frames(self) -> List[Dict[str, Any]]:
        """Get all frame information."""
        return self.transforms["frames"]
    
    def get_cameras(self) -> Dict[str, Any]:
        """Get camera calibrations."""
        cameras = {}
        for frame in self.transforms["frames"]:
            cam = frame.get("camera", "default")
            if cam not in cameras:
                cameras[cam] = {
                    "fl_x": frame["fl_x"],
                    "fl_y": frame["fl_y"],
                    "cx": frame["cx"],
                    "cy": frame["cy"],
                    "w": frame["w"],
                    "h": frame["h"]
                }
        return cameras
    
    def get_time_range(self) -> Tuple[float, float]:
        """Get time range of the sequence."""
        times = [frame["time"] for frame in self.transforms["frames"]]
        return min(times), max(times)


def process_video_sequence(
    input_path: Union[str, List[str]],
    output_dir: str,
    calibration: Optional[Union[str, Dict[str, Any]]] = None,
    target_fps: float = 30.0,
    resize: Optional[Tuple[int, int]] = None,
    extract_every_n: int = 1
) -> str:
    """
    High-level function to process video sequence(s).
    
    Args:
        input_path: Video file, directory of videos, or list of video paths
        output_dir: Output directory for processed data
        calibration: Calibration file or dictionary
        target_fps: Target FPS
        resize: Optional resize dimensions
        extract_every_n: Extract every N frames
        
    Returns:
        Path to output directory with processed data
    """
    processor = VideoProcessor(
        output_dir=output_dir,
        target_fps=target_fps,
        resize=resize,
        extract_every_n=extract_every_n
    )
    
    # Handle different input types
    if isinstance(input_path, list):
        # Multiple video files
        cameras = []
        for idx, video_path in enumerate(input_path):
            cameras.append(CameraConfig(
                name=f"cam{idx}",
                video_path=video_path,
                calibration=calibration.get(f"cam{idx}") if isinstance(calibration, dict) else None
            ))
        metadata = processor.process_multi_camera_videos(cameras)
        
    elif os.path.isdir(input_path):
        # Directory of videos
        metadata = processor.process_video_directory(
            input_path,
            calibration_file=calibration if isinstance(calibration, str) else None
        )
        
    else:
        # Single video file
        metadata = processor.process_single_camera_video(
            input_path,
            calibration=calibration if isinstance(calibration, dict) else None
        )
    
    # Create transforms.json for our dataloader
    processor.create_transforms_json(metadata)
    
    logger.info(f"Processing complete. Output saved to {output_dir}")
    return output_dir