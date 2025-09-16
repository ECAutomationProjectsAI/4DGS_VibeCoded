"""Base converter class for 4D Gaussian Splatting formats."""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from pathlib import Path


class BaseConverter(ABC):
    """Abstract base class for 4DGS format converters."""
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize converter.
        
        Args:
            chunk_size: Number of Gaussians to process at once (for memory efficiency)
        """
        self.chunk_size = chunk_size
    
    @abstractmethod
    def load(self, path: str) -> Dict[str, Any]:
        """
        Load 4DGS data from external format.
        
        Args:
            path: Path to the external format data
            
        Returns:
            Dictionary containing:
                - xyz: (N, 3) positions
                - vel: (N, 3) velocities (optional)
                - t: (N, 1) time centers
                - scales: (N, 3) scales
                - scale_t: (N, 1) temporal scales
                - rotations: (N, 4) quaternions
                - opacity: (N, 1) opacities
                - features: (N, 3, C) SH coefficients or RGB
                - metadata: Dict with additional info
        """
        pass
    
    @abstractmethod
    def save(self, data: Dict[str, Any], path: str) -> None:
        """
        Save 4DGS data to external format.
        
        Args:
            data: Dictionary with 4DGS data
            path: Output path
        """
        pass
    
    def convert_to_gs4d(self, input_path: str, output_path: str) -> None:
        """
        Convert from external format to our gs4d format.
        
        Args:
            input_path: Path to external format
            output_path: Path to save gs4d checkpoint
        """
        data = self.load(input_path)
        
        # Convert to gs4d state dict format
        state = {
            'xyz': data['xyz'],
            'vel': data.get('vel', torch.zeros_like(data['xyz'])),
            't': data.get('t', torch.zeros(data['xyz'].shape[0], 1)),
            'log_scale': torch.log(data['scales']),
            'log_scale_t': torch.log(data.get('scale_t', torch.ones(data['xyz'].shape[0], 1) * 0.1)),
            'quat': data['rotations'],
            'omega_t': data.get('omega_t', torch.zeros(data['xyz'].shape[0], 1)),
            'opacity': data['opacity'],
            'rgb_sh': data['features']
        }
        
        # Save checkpoint
        torch.save({'state': state, 'metadata': data.get('metadata', {})}, output_path)
        print(f"Converted to gs4d format: {output_path}")
    
    def convert_from_gs4d(self, input_path: str, output_path: str) -> None:
        """
        Convert from gs4d format to external format.
        
        Args:
            input_path: Path to gs4d checkpoint
            output_path: Path to save in external format
        """
        checkpoint = torch.load(input_path, map_location='cpu')
        state = checkpoint['state']
        
        # Convert from gs4d state dict
        data = {
            'xyz': state['xyz'],
            'vel': state.get('vel', torch.zeros_like(state['xyz'])),
            't': state.get('t', torch.zeros(state['xyz'].shape[0], 1)),
            'scales': torch.exp(state['log_scale']),
            'scale_t': torch.exp(state.get('log_scale_t', torch.zeros(state['xyz'].shape[0], 1))),
            'rotations': state['quat'],
            'opacity': state['opacity'],
            'features': state['rgb_sh'],
            'metadata': checkpoint.get('metadata', {})
        }
        
        self.save(data, output_path)
        print(f"Converted from gs4d format to {self.__class__.__name__}")
    
    @staticmethod
    def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
        """Normalize quaternions to unit length."""
        return q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    
    @staticmethod
    def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices."""
        q = BaseConverter.normalize_quaternion(q)
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        R = torch.zeros(q.shape[:-1] + (3, 3), device=q.device)
        
        R[..., 0, 0] = 1 - 2 * (y**2 + z**2)
        R[..., 0, 1] = 2 * (x * y - w * z)
        R[..., 0, 2] = 2 * (x * z + w * y)
        
        R[..., 1, 0] = 2 * (x * y + w * z)
        R[..., 1, 1] = 1 - 2 * (x**2 + z**2)
        R[..., 1, 2] = 2 * (y * z - w * x)
        
        R[..., 2, 0] = 2 * (x * z - w * y)
        R[..., 2, 1] = 2 * (y * z + w * x)
        R[..., 2, 2] = 1 - 2 * (x**2 + y**2)
        
        return R
    
    @staticmethod
    def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to quaternions."""
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        
        q = torch.zeros(R.shape[:-2] + (4,), device=R.device)
        
        # Compute quaternion components
        s = 0.5 / torch.sqrt(trace + 1.0 + 1e-8)
        q[..., 0] = 0.25 / s
        q[..., 1] = (R[..., 2, 1] - R[..., 1, 2]) * s
        q[..., 2] = (R[..., 0, 2] - R[..., 2, 0]) * s
        q[..., 3] = (R[..., 1, 0] - R[..., 0, 1]) * s
        
        return BaseConverter.normalize_quaternion(q)