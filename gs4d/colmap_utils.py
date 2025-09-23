"""COLMAP utility functions for 4DGS pipeline (RunPod/Ubuntu).

Provides simple, robust helpers to:
- run_colmap_sfm: feature extraction, matching, mapper, and model conversion to TXT
- parse_colmap_output: read cameras.txt and images.txt
- quaternion_to_matrix: convert COLMAP quaternion to rotation matrix

This isolates COLMAP logic from legacy preprocessing scripts and keeps separation of concerns.
"""

from __future__ import annotations
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float):
    import numpy as np
    R = np.zeros((3, 3), dtype=np.float32)
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


def run_colmap_sfm(
    image_dir: str,
    colmap_dir: str,
    camera_model: str = "OPENCV",
    threads: int = 0,
    max_image_size: int = 0,
    max_num_features: int = 0,
    matcher: str = "exhaustive",
    single_camera: bool = False,
) -> bool:
    """
    Run COLMAP SfM pipeline headlessly, writing logs under colmap_dir/logs and sparse model to colmap_dir.
    """
    image_dir = str(image_dir)
    colmap_dir = Path(colmap_dir)
    colmap_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = colmap_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    database_path = colmap_dir / "database.db"

    # Clean database if exists (fresh run)
    try:
        if database_path.exists():
            database_path.unlink()
    except Exception:
        pass

    # Headless env
    colmap_env = os.environ.copy()
    colmap_env.pop('QT_PLUGIN_PATH', None)
    colmap_env.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
    colmap_env['QT_QPA_PLATFORM'] = 'offscreen'
    if threads and threads > 0:
        colmap_env.setdefault('OMP_NUM_THREADS', str(threads))

    def _run_and_stream(cmd: List[str], log_name: str) -> int:
        log_path = logs_dir / log_name
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

    def _colmap_has_option(subcmd: str, option: str) -> bool:
        try:
            hp = subprocess.run(["colmap", subcmd, "-h"], check=False, capture_output=True, text=True, env=colmap_env)
            return option in (hp.stdout or "")
        except Exception:
            return False

    # 1) Feature extraction
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", image_dir,
        "--ImageReader.camera_model", camera_model
    ]
    if _colmap_has_option("feature_extractor", "--SiftExtraction.use_gpu"):
        cmd.extend(["--SiftExtraction.use_gpu", "0"])  # safer in headless
    if threads > 0 and _colmap_has_option("feature_extractor", "--SiftExtraction.num_threads"):
        cmd.extend(["--SiftExtraction.num_threads", str(threads)])
    if max_image_size > 0 and _colmap_has_option("feature_extractor", "--SiftExtraction.max_image_size"):
        cmd.extend(["--SiftExtraction.max_image_size", str(max_image_size)])
    if max_num_features > 0 and _colmap_has_option("feature_extractor", "--SiftExtraction.max_num_features"):
        cmd.extend(["--SiftExtraction.max_num_features", str(max_num_features)])
    if single_camera:
        cmd.extend(["--ImageReader.single_camera", "1"])
    ret = _run_and_stream(cmd, "01_feature_extractor.log")
    if ret != 0:
        logger.error(f"Feature extraction failed with code {ret}")
        return False

    # 2) Matching
    if matcher == "sequential":
        cmd = ["colmap", "sequential_matcher", "--database_path", str(database_path), "--SequentialMatching.overlap", "5"]
    else:
        cmd = ["colmap", "exhaustive_matcher", "--database_path", str(database_path)]
    if _colmap_has_option(cmd[1], "--SiftMatching.use_gpu"):
        cmd.extend(["--SiftMatching.use_gpu", "0"])  # safer
    ret = _run_and_stream(cmd, "02_matcher.log")
    if ret != 0:
        logger.error(f"Feature matching failed with code {ret}")
        return False

    # 3) Mapper
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", image_dir,
        "--output_path", str(sparse_dir)
    ]
    ret = _run_and_stream(cmd, "03_mapper.log")
    if ret != 0:
        logger.error(f"Sparse reconstruction failed with code {ret}")
        return False

    # 4) Model converter -> TXT
    sparse_txt_dir = colmap_dir / "sparse_txt"
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


def parse_colmap_output(colmap_dir: str) -> Dict:
    """Parse cameras.txt and images.txt from colmap_dir/sparse_txt."""
    import numpy as np
    colmap_dir = Path(colmap_dir)
    sparse_txt = colmap_dir / "sparse_txt"

    cameras = {}
    cameras_file = sparse_txt / "cameras.txt"
    if cameras_file.exists():
        with open(cameras_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
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
                        'params': params,
                    }

    images = {}
    images_file = sparse_txt / "images.txt"
    if images_file.exists():
        with open(images_file, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if lines[i].startswith('#'):
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
                        'camera_id': camera_id,
                    }

    return {'cameras': cameras, 'images': images}
