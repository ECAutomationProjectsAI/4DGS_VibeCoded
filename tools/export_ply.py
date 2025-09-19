"""
Export 4D Gaussian Splatting models to PLY format.

Supports:
- Single PLY file with all temporal information
- Sequence of PLY files (one per frame)
- Compatible with standard Gaussian Splatting viewers
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Add parent directory to path to import gs4d module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs4d.gaussians import GaussianModel4D
from gs4d.dataio import load_transforms


def construct_list_of_attributes():
    """Define PLY attributes for 4D Gaussians."""
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']  # Position and normal (velocity)
    # Add temporal attributes
    l.append('t')  # Time center
    l.append('scale_t')  # Temporal scale
    
    # Spatial scales
    for i in range(3):
        l.append(f'scale_{i}')
    
    # Rotation (quaternion)
    for i in range(4):
        l.append(f'rot_{i}')
    
    # Spherical harmonics coefficients
    l.append('f_dc_0')
    l.append('f_dc_1')
    l.append('f_dc_2')
    
    # Higher order SH
    for i in range(45):  # 3rd degree SH has 45 additional coefficients (3*(degree+1)^2 - 3)
        l.append(f'f_rest_{i}')
    
    # Opacity
    l.append('opacity')
    
    return l


def save_ply(xyz: np.ndarray, 
             features: Dict[str, np.ndarray],
             path: str,
             include_normals: bool = True):
    """
    Save Gaussians to PLY format.
    
    Args:
        xyz: [N, 3] positions
        features: Dictionary of features to save
        path: Output path
        include_normals: Whether to include velocity as normals
    """
    import struct
    
    # Prepare vertex data
    normals = features.get('normals', np.zeros_like(xyz))
    
    # Construct attributes list
    attrs = []
    attrs.extend(['x', 'y', 'z'])
    if include_normals:
        attrs.extend(['nx', 'ny', 'nz'])
    
    # Add all other features
    for key in ['t', 'scale_t', 'scale_0', 'scale_1', 'scale_2',
                'rot_0', 'rot_1', 'rot_2', 'rot_3',
                'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity']:
        if key in features:
            attrs.append(key)
    
    # Add f_rest features
    for i in range(45):
        key = f'f_rest_{i}'
        if key in features:
            attrs.append(key)
    
    # Create dtype list
    dtype_full = [(attr, 'f4') for attr in attrs]
    
    # Create structured array
    N = xyz.shape[0]
    elements = np.empty(N, dtype=dtype_full)
    
    # Fill position
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    
    # Fill normals if included
    if include_normals:
        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]
    
    # Fill other features
    for key, value in features.items():
        if key == 'normals':
            continue
        if key in attrs:
            elements[key] = value.flatten()
    
    # Write PLY header
    with open(path, 'wb') as f:
        # Header
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {N}\n'.encode('ascii'))
        
        for attr in attrs:
            f.write(f'property float {attr}\n'.encode('ascii'))
        
        f.write(b'end_header\n')
        
        # Write binary data
        f.write(elements.tobytes())
    
    print(f"Saved {N} Gaussians to {path}")


def export_single_ply(model: GaussianModel4D, 
                     output_path: str,
                     time: float = 0.0):
    """
    Export model to a single PLY file at specified time.
    
    Args:
        model: Trained 4DGS model
        output_path: Output PLY file path
        time: Time to evaluate (default 0.0 for center time)
    """
    # Get positions at specified time
    xyz = model.position_at_time(time).detach().cpu().numpy()
    
    # Prepare features
    features = {}
    
    # Velocity as normals
    features['normals'] = model.vel.detach().cpu().numpy()
    
    # Temporal attributes
    features['t'] = model.t.detach().cpu().numpy().squeeze()
    features['scale_t'] = model.scale_t.detach().cpu().numpy().squeeze()
    
    # Spatial scales
    scales = model.scales.detach().cpu().numpy()
    for i in range(3):
        features[f'scale_{i}'] = scales[:, i]
    
    # Rotation quaternion
    quat = model.quat.detach().cpu().numpy()
    for i in range(4):
        features[f'rot_{i}'] = quat[:, i]
    
    # SH coefficients
    sh_coeffs = model.rgb_sh.detach().cpu().numpy()  # [N, 3, C]
    N, _, C = sh_coeffs.shape
    
    # DC coefficients (0th degree)
    for i in range(3):
        features[f'f_dc_{i}'] = sh_coeffs[:, i, 0]
    
    # Higher order coefficients
    if C > 1:
        sh_rest = sh_coeffs[:, :, 1:].reshape(N, -1)  # Flatten remaining coeffs
        for i in range(min(45, sh_rest.shape[1])):
            features[f'f_rest_{i}'] = sh_rest[:, i]
    
    # Opacity
    features['opacity'] = model.opacity.detach().cpu().numpy().squeeze()
    
    # Save PLY
    save_ply(xyz, features, output_path)


def export_ply_sequence(model: GaussianModel4D,
                       output_dir: str,
                       num_frames: int = 100,
                       time_range: tuple = (-0.5, 0.5)):
    """
    Export model as a sequence of PLY files.
    
    Args:
        model: Trained 4DGS model
        output_dir: Output directory for PLY sequence
        num_frames: Number of frames to generate
        time_range: (min_time, max_time) range
    """
    os.makedirs(output_dir, exist_ok=True)
    
    times = np.linspace(time_range[0], time_range[1], num_frames)
    
    print(f"Exporting {num_frames} PLY files to {output_dir}")
    for i, t in enumerate(tqdm(times, desc="Exporting frames")):
        output_path = os.path.join(output_dir, f"frame_{i:05d}.ply")
        export_single_ply(model, output_path, time=t)
    
    # Create metadata file
    metadata = {
        "num_frames": num_frames,
        "time_range": list(time_range),
        "fps": 30,  # Default playback FPS
        "format": "4dgs_sequence"
    }
    
    import json
    with open(os.path.join(output_dir, "sequence_info.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Sequence exported successfully!")
    print(f"Metadata saved to sequence_info.json")


def export_4dgs_format(model: GaussianModel4D,
                      output_path: str):
    """
    Export to custom 4DGS format with full temporal information.
    This format can be loaded directly for 4D playback.
    
    Args:
        model: Trained 4DGS model
        output_path: Output file path (.4dgs extension recommended)
    """
    # Prepare all model data
    data = {
        'version': '1.0',
        'type': '4dgs',
        'num_points': model.primitive.n_points(),
        'sh_degree': model.primitive.sh_degree,
        'data': {
            'xyz': model.xyz.detach().cpu().numpy(),
            'vel': model.vel.detach().cpu().numpy(),
            't': model.t.detach().cpu().numpy(),
            'log_scale': model.primitive._log_scale.detach().cpu().numpy(),
            'log_scale_t': model.primitive._log_scale_t.detach().cpu().numpy(),
            'quat': model.quat.detach().cpu().numpy(),
            'opacity': model.primitive._opacity.detach().cpu().numpy(),
            'rgb_sh': model.rgb_sh.detach().cpu().numpy(),
        }
    }
    
    # Add optional parameters if they exist
    if hasattr(model.primitive, '_omega_t') and model.primitive._omega_t is not None:
        data['data']['omega_t'] = model.primitive._omega_t.detach().cpu().numpy()
    
    # Save using numpy's compressed format
    np.savez_compressed(output_path, **data)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved 4DGS format to {output_path} ({file_size_mb:.2f} MB)")
    print(f"  Points: {data['num_points']:,}")
    print(f"  SH Degree: {data['sh_degree']}")


def main():
    parser = argparse.ArgumentParser(description='Export 4DGS model to PLY or custom format')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--output', type=str, required=True, help='Output path (file or directory)')
    parser.add_argument('--format', type=str, default='ply', 
                       choices=['ply', 'ply_sequence', '4dgs'],
                       help='Export format')
    parser.add_argument('--num_frames', type=int, default=100,
                       help='Number of frames for sequence export')
    parser.add_argument('--time', type=float, default=0.0,
                       help='Time value for single PLY export')
    parser.add_argument('--time_min', type=float, default=-0.5,
                       help='Minimum time for sequence')
    parser.add_argument('--time_max', type=float, default=0.5,
                       help='Maximum time for sequence')
    parser.add_argument('--sh_degree', type=int, default=3,
                       help='SH degree of the model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.ckpt):
        print(f"ERROR: Checkpoint not found: {args.ckpt}")
        sys.exit(1)
    
    # Load model
    print(f"Loading checkpoint from {args.ckpt}...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt['state']
    
    model = GaussianModel4D(sh_degree=args.sh_degree, device=device)
    model.load_state_dict_compact(state)
    
    print(f"Model loaded: {model.primitive.n_points():,} Gaussians")
    
    # Export based on format
    if args.format == 'ply':
        # Single PLY at specified time
        export_single_ply(model, args.output, time=args.time)
        
    elif args.format == 'ply_sequence':
        # Sequence of PLY files
        export_ply_sequence(
            model, 
            args.output,
            num_frames=args.num_frames,
            time_range=(args.time_min, args.time_max)
        )
        
    elif args.format == '4dgs':
        # Custom 4DGS format with full temporal data
        if not args.output.endswith('.4dgs'):
            args.output += '.4dgs'
        export_4dgs_format(model, args.output)
    
    print("Export complete!")


if __name__ == '__main__':
    main()