"""SpacetimeGaussian format converter.

Handles conversion to/from SpacetimeGaussian PLY format with timestamps.
Reference: https://github.com/oppo-us-research/SpacetimeGaussian
"""

import os
import struct
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from .base import BaseConverter


class SpacetimeGaussianConverter(BaseConverter):
    """Converter for SpacetimeGaussian PLY format."""
    
    def __init__(self, chunk_size: int = 10000, sh_degree: int = 3):
        """
        Initialize SpacetimeGaussian converter.
        
        Args:
            chunk_size: Number of Gaussians to process at once
            sh_degree: Spherical harmonics degree (0-3)
        """
        super().__init__(chunk_size)
        self.sh_degree = sh_degree
        self.sh_channels = (sh_degree + 1) ** 2
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        Load SpacetimeGaussian PLY file.
        
        SpacetimeGaussian format typically includes:
        - x, y, z: 3D positions
        - t: timestamp
        - scale_0, scale_1, scale_2: 3D scales
        - rot_0, rot_1, rot_2, rot_3: quaternion rotation
        - opacity: opacity value
        - f_dc_0, f_dc_1, f_dc_2: DC spherical harmonics
        - f_rest_*: Higher order SH coefficients
        - motion_*: Optional motion/deformation parameters
        """
        plydata = self._read_ply(path)
        
        # Extract basic properties
        xyz = np.stack([
            plydata['x'],
            plydata['y'], 
            plydata['z']
        ], axis=1)
        
        # Time information
        if 't' in plydata:
            t = plydata['t'].reshape(-1, 1)
        elif 'timestamp' in plydata:
            t = plydata['timestamp'].reshape(-1, 1)
        else:
            # Assume static if no time info
            t = np.zeros((xyz.shape[0], 1))
        
        # Scales
        scales = np.stack([
            plydata['scale_0'],
            plydata['scale_1'],
            plydata['scale_2']
        ], axis=1)
        
        # Rotations (quaternion)
        rotations = np.stack([
            plydata.get('rot_0', np.ones(len(xyz))),  # w
            plydata.get('rot_1', np.zeros(len(xyz))), # x
            plydata.get('rot_2', np.zeros(len(xyz))), # y
            plydata.get('rot_3', np.zeros(len(xyz)))  # z
        ], axis=1)
        
        # Opacity
        opacity = plydata['opacity'].reshape(-1, 1)
        
        # SH features
        features_dc = np.stack([
            plydata['f_dc_0'],
            plydata['f_dc_1'],
            plydata['f_dc_2']
        ], axis=1).reshape(-1, 3, 1)
        
        # Collect higher order SH
        features_rest = []
        for i in range(1, self.sh_channels):
            if f'f_rest_{i*3}' in plydata:
                feat = np.stack([
                    plydata[f'f_rest_{i*3}'],
                    plydata[f'f_rest_{i*3+1}'],
                    plydata[f'f_rest_{i*3+2}']
                ], axis=1).reshape(-1, 3, 1)
                features_rest.append(feat)
        
        if features_rest:
            features_rest = np.concatenate(features_rest, axis=2)
            features = np.concatenate([features_dc, features_rest], axis=2)
        else:
            features = features_dc
        
        # Check for velocity/motion fields
        vel = None
        if 'vel_x' in plydata:
            vel = np.stack([
                plydata['vel_x'],
                plydata['vel_y'],
                plydata['vel_z']
            ], axis=1)
        elif 'motion_x' in plydata:
            vel = np.stack([
                plydata['motion_x'],
                plydata['motion_y'],
                plydata['motion_z']
            ], axis=1)
        
        # Temporal scale
        scale_t = plydata.get('scale_t', np.ones((len(xyz), 1)) * 0.1).reshape(-1, 1)
        
        # Convert to torch tensors
        data = {
            'xyz': torch.from_numpy(xyz).float(),
            't': torch.from_numpy(t).float(),
            'scales': torch.from_numpy(np.exp(scales)).float(),  # SpacetimeGaussian stores log scales
            'scale_t': torch.from_numpy(np.exp(scale_t)).float() if 'scale_t' in plydata else torch.ones(len(xyz), 1) * 0.1,
            'rotations': torch.from_numpy(rotations).float(),
            'opacity': torch.from_numpy(opacity).float(),
            'features': torch.from_numpy(features).float(),
            'metadata': {
                'source_format': 'SpacetimeGaussian',
                'sh_degree': self.sh_degree,
                'num_points': len(xyz)
            }
        }
        
        if vel is not None:
            data['vel'] = torch.from_numpy(vel).float()
        
        return data
    
    def save(self, data: Dict[str, Any], path: str) -> None:
        """Save data in SpacetimeGaussian PLY format."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Convert tensors to numpy
        xyz = data['xyz'].cpu().numpy()
        t = data.get('t', np.zeros((len(xyz), 1))).cpu().numpy()
        scales = np.log(data['scales'].cpu().numpy())  # Convert to log scale
        rotations = data['rotations'].cpu().numpy()
        opacity = data['opacity'].cpu().numpy()
        features = data['features'].cpu().numpy()
        
        # Prepare PLY data
        plydata = {
            'x': xyz[:, 0],
            'y': xyz[:, 1],
            'z': xyz[:, 2],
            't': t.flatten(),
            'scale_0': scales[:, 0],
            'scale_1': scales[:, 1],
            'scale_2': scales[:, 2],
            'rot_0': rotations[:, 0],
            'rot_1': rotations[:, 1],
            'rot_2': rotations[:, 2],
            'rot_3': rotations[:, 3],
            'opacity': opacity.flatten()
        }
        
        # Add SH features
        plydata['f_dc_0'] = features[:, 0, 0]
        plydata['f_dc_1'] = features[:, 1, 0]
        plydata['f_dc_2'] = features[:, 2, 0]
        
        # Add higher order SH if present
        if features.shape[2] > 1:
            for i in range(1, features.shape[2]):
                plydata[f'f_rest_{i*3}'] = features[:, 0, i]
                plydata[f'f_rest_{i*3+1}'] = features[:, 1, i]
                plydata[f'f_rest_{i*3+2}'] = features[:, 2, i]
        
        # Add velocity if present
        if 'vel' in data:
            vel = data['vel'].cpu().numpy()
            plydata['vel_x'] = vel[:, 0]
            plydata['vel_y'] = vel[:, 1]
            plydata['vel_z'] = vel[:, 2]
        
        # Add temporal scale
        if 'scale_t' in data:
            scale_t = np.log(data['scale_t'].cpu().numpy())
            plydata['scale_t'] = scale_t.flatten()
        
        self._write_ply(path, plydata)
    
    def _read_ply(self, path: str) -> Dict[str, np.ndarray]:
        """Read PLY file and return properties as dict."""
        with open(path, 'rb') as f:
            # Read header
            header = []
            while True:
                line = f.readline().decode('utf-8').strip()
                header.append(line)
                if line == 'end_header':
                    break
            
            # Parse header
            vertex_count = 0
            properties = []
            for line in header:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('property'):
                    parts = line.split()
                    dtype = parts[1]
                    name = parts[2]
                    properties.append((name, self._ply_dtype_to_numpy(dtype)))
            
            # Read binary data
            data = {}
            for prop_name, dtype in properties:
                values = np.fromfile(f, dtype=dtype, count=vertex_count)
                data[prop_name] = values
            
            return data
    
    def _write_ply(self, path: str, data: Dict[str, np.ndarray]) -> None:
        """Write PLY file from properties dict."""
        # Get vertex count from first property
        vertex_count = len(next(iter(data.values())))
        
        with open(path, 'wb') as f:
            # Write header
            f.write(b'ply\n')
            f.write(b'format binary_little_endian 1.0\n')
            f.write(f'element vertex {vertex_count}\n'.encode())
            
            # Write property definitions
            for prop_name in data.keys():
                dtype = data[prop_name].dtype
                ply_type = self._numpy_dtype_to_ply(dtype)
                f.write(f'property {ply_type} {prop_name}\n'.encode())
            
            f.write(b'end_header\n')
            
            # Write binary data
            for i in range(vertex_count):
                for prop_name in data.keys():
                    value = data[prop_name][i]
                    f.write(value.tobytes())
    
    def _ply_dtype_to_numpy(self, ply_dtype: str) -> np.dtype:
        """Convert PLY data type to numpy dtype."""
        mapping = {
            'float': np.float32,
            'double': np.float64,
            'uchar': np.uint8,
            'char': np.int8,
            'ushort': np.uint16,
            'short': np.int16,
            'uint': np.uint32,
            'int': np.int32
        }
        return mapping.get(ply_dtype, np.float32)
    
    def _numpy_dtype_to_ply(self, numpy_dtype: np.dtype) -> str:
        """Convert numpy dtype to PLY data type."""
        mapping = {
            np.float32: 'float',
            np.float64: 'double',
            np.uint8: 'uchar',
            np.int8: 'char',
            np.uint16: 'ushort',
            np.int16: 'short',
            np.uint32: 'uint',
            np.int32: 'int'
        }
        for key, value in mapping.items():
            if numpy_dtype == key:
                return value
        return 'float'