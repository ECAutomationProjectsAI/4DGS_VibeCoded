import os
import json
import argparse
import shutil
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

"""
Convert a nerfstudio-style transforms.json dataset (e.g., Diffuman4D outputs) into
a COLMAP-style folder compatible with SpacetimeGaussian/3DGS loaders.

Output layout:
<dst>/colmap_0/
  images/               # copied images
  sparse/0/
    cameras.txt
    images.txt
    points3D.txt        # optional (empty if no sparse_pcd.ply)

Assumptions:
- transforms.json is nerfstudio/OpenGL (Blender-like) c2w convention.
- We convert to COLMAP/OpenCV via left-multiplying diag([1,-1,-1,1]).
- We write PINHOLE intrinsics (fx, fy, cx, cy).
- If frames include "camera_label" we use per-label camera IDs; else single camera ID.
- If sparse_pcd.ply exists (ascii), we convert to points3D.txt; otherwise points3D.txt is empty.

Usage example:
python tools/convert_transforms_to_colmap.py --src data/synth_tiny --dst out_scene --copy_images
"""

T_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def mat_to_qvec_tvec(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert rotation matrix to quaternion [qw, qx, qy, qz] and translation t.
    R is world->cam.
    """
    # Ensure proper orthonormalization (optional small fix)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    qw = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    qx = np.sign(R[2, 1] - R[1, 2]) * np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    qy = np.sign(R[0, 2] - R[2, 0]) * np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    qz = np.sign(R[1, 0] - R[0, 1]) * np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    # Normalize
    q = q / (np.linalg.norm(q) + 1e-12)
    return q, t.astype(np.float64)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_intrinsics(meta: Dict[str, Any]) -> Tuple[float, float, float, float, int, int]:
    H, W = int(meta['h']), int(meta['w'])
    fx, fy = float(meta['fl_x']), float(meta['fl_y'])
    cx, cy = float(meta['cx']), float(meta['cy'])
    return fx, fy, cx, cy, H, W


def copy_images(frames: List[Dict[str, Any]], src_root: str, images_out_dir: str):
    for fr in frames:
        src = os.path.join(src_root, fr['file_path'])
        dst = os.path.join(images_out_dir, os.path.basename(fr['file_path']))
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)
        fr['colmap_name'] = os.path.basename(fr['file_path'])


def build_camera_map(frames: List[Dict[str, Any]]) -> Dict[str, int]:
    cam_labels = []
    for fr in frames:
        cam_labels.append(fr.get('camera_label', 'default'))
    uniq = sorted(set(cam_labels))
    return {label: (i + 1) for i, label in enumerate(uniq)}  # CAMERA_ID starts at 1


def write_cameras_txt(path: str, camera_map: Dict[str, int], intrinsics: Tuple[float, float, float, float, int, int]):
    fx, fy, cx, cy, H, W = intrinsics
    lines = []
    lines.append('# Camera list with one line of data per camera:')
    lines.append('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]')
    for label, cam_id in camera_map.items():
        lines.append(f'{cam_id} PINHOLE {W} {H} {fx} {fy} {cx} {cy}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_images_txt(path: str, frames: List[Dict[str, Any]], camera_map: Dict[str, int], meta_is_gl: bool = True):
    lines = []
    lines.append('# Image list with two lines of data per image:')
    lines.append('#   IMAGE_ID, QW QX QY QZ, TX TY TZ, CAMERA_ID, NAME')
    lines.append('#   POINTS2D[] as (X, Y, POINT3D_ID)')
    image_id = 1
    for fr in frames:
        c2w = np.array(fr['transform_matrix'], dtype=np.float32)  # OpenGL/Blender
        if meta_is_gl:
            c2w_cv = T_GL2CV @ c2w
        else:
            c2w_cv = c2w
        R_c2w = c2w_cv[:3, :3]
        t_c2w = c2w_cv[:3, 3]
        # World->Cam
        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w
        q, tvec = mat_to_qvec_tvec(R_w2c, t_w2c)
        cam_id = camera_map.get(fr.get('camera_label', 'default'), 1)
        name = fr.get('colmap_name', os.path.basename(fr['file_path']))
        lines.append(f'{image_id} {q[0]} {q[1]} {q[2]} {q[3]} {tvec[0]} {tvec[1]} {tvec[2]} {cam_id} {name}')
        lines.append('')  # no 2D points
        image_id += 1
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_points3D_txt(path: str, ply_path: Optional[str]):
    lines = []
    lines.append('# 3D point list with one line of data per point:')
    lines.append('#   POINT3D_ID, X Y Z, R G B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)')
    wrote_any = False
    if ply_path and os.path.isfile(ply_path):
        try:
            with open(ply_path, 'rb') as f:
                header = []
                while True:
                    line = f.readline()
                    if not line:
                        break
                    header.append(line)
                    if line.strip() == b'end_header':
                        break
                header_text = b''.join(header).decode('utf-8', errors='ignore')
                if 'format ascii' in header_text:
                    # Restart reading as text
                    with open(ply_path, 'r') as ft:
                        hdr = []
                        for l in ft:
                            hdr.append(l)
                            if l.strip() == 'end_header':
                                break
                        for idx, l in enumerate(ft, start=1):
                            parts = l.strip().split()
                            if len(parts) < 3:
                                continue
                            x, y, z = map(float, parts[:3])
                            r, g, b = (255, 255, 255)
                            if len(parts) >= 6:
                                try:
                                    r, g, b = map(int, parts[3:6])
                                except Exception:
                                    pass
                            lines.append(f'{idx} {x} {y} {z} {r} {g} {b} 1.0')
                            wrote_any = True
                else:
                    # Unsupported binary ply in this minimal script
                    pass
        except Exception:
            pass
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))


def main():
    ap = argparse.ArgumentParser(description='Convert nerfstudio transforms.json to COLMAP text format')
    ap.add_argument('--src', required=True, help='Source sequence root containing transforms.json and frames/images')
    ap.add_argument('--dst', required=True, help='Destination root where colmap_0 will be created')
    ap.add_argument('--assume_opengl', action='store_true', default=True, help='If transforms.json is in OpenGL/Blender coords (default True)')
    ap.add_argument('--copy_images', action='store_true', default=True, help='Copy images into colmap_0/images (default True)')
    ap.add_argument('--sparse_pcd', type=str, default=None, help='Optional path to sparse_pcd.ply to convert into points3D.txt')
    args = ap.parse_args()

    meta = load_json(os.path.join(args.src, 'transforms.json'))
    frames = meta['frames']

    fx, fy, cx, cy, H, W = parse_intrinsics(meta)

    colmap_root = os.path.join(args.dst, 'colmap_0')
    images_out = os.path.join(colmap_root, 'images')
    sparse_out = os.path.join(colmap_root, 'sparse', '0')
    ensure_dir(images_out)
    ensure_dir(sparse_out)

    if args.copy_images:
        copy_images(frames, args.src, images_out)
    else:
        # Just set colmap_name without copying
        for fr in frames:
            fr['colmap_name'] = os.path.basename(fr['file_path'])

    cam_map = build_camera_map(frames)

    write_cameras_txt(os.path.join(sparse_out, 'cameras.txt'), cam_map, (fx, fy, cx, cy, H, W))
    write_images_txt(os.path.join(sparse_out, 'images.txt'), frames, cam_map, meta_is_gl=args.assume_opengl)
    # points3D
    ply_path = args.sparse_pcd if args.sparse_pcd else os.path.join(args.src, 'sparse_pcd.ply')
    write_points3D_txt(os.path.join(sparse_out, 'points3D.txt'), ply_path if os.path.isfile(ply_path) else None)

    print(f'Wrote COLMAP text scene to: {colmap_root}')


if __name__ == '__main__':
    main()
