"""Fudan 4DGS format converter.

Handles conversion to/from Fudan University's 4D Gaussian Splatting format.
This format uses deformation fields and neural networks for dynamic modeling.
Reference: https://github.com/fudan-zvg/4d-gaussian-splatting
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from .base import BaseConverter


class FudanConverter(BaseConverter):
    """Converter for Fudan 4DGS format with deformation fields."""
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize Fudan 4DGS converter.
        
        Args:
            chunk_size: Number of Gaussians to process at once
        """
        super().__init__(chunk_size)
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        Load Fudan 4DGS checkpoint.
        
        Fudan format typically includes:
        - Canonical Gaussians (static representation)
        - Deformation network weights
        - Time-dependent deformations
        - Possibly multiple timestamps
        """
        if path.endswith('.pth') or path.endswith('.pt'):
            checkpoint = torch.load(path, map_location='cpu')
        else:
            # Try to load from directory
            checkpoint_path = os.path.join(path, 'checkpoint.pth')
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(path, 'model.pth')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract Gaussian parameters
        if 'gaussians' in checkpoint:
            gaussian_params = checkpoint['gaussians']
        elif 'model_state_dict' in checkpoint:
            gaussian_params = self._extract_gaussian_params(checkpoint['model_state_dict'])
        else:
            gaussian_params = checkpoint
        
        # Get canonical Gaussians
        xyz = gaussian_params.get('_xyz', gaussian_params.get('xyz', None))
        if xyz is None:
            raise ValueError("No position data found in checkpoint")
        
        # Get other parameters
        features_dc = gaussian_params.get('_features_dc', gaussian_params.get('features_dc', None))
        features_rest = gaussian_params.get('_features_rest', gaussian_params.get('features_rest', None))
        scaling = gaussian_params.get('_scaling', gaussian_params.get('scaling', None))
        rotation = gaussian_params.get('_rotation', gaussian_params.get('rotation', None))
        opacity = gaussian_params.get('_opacity', gaussian_params.get('opacity', None))
        
        # Handle deformation if present
        deformation = None
        if 'deformation' in checkpoint:
            deformation = checkpoint['deformation']
        elif 'deformation_network' in checkpoint:
            deformation = checkpoint['deformation_network']
        
        # Convert scales from log if needed
        if scaling is not None:
            # Fudan often stores log scales
            if scaling.min() < -10:  # Likely log scale
                scales = torch.exp(scaling)
            else:
                scales = scaling
        else:
            scales = torch.ones(xyz.shape[0], 3) * 0.01
        
        # Convert rotation to quaternions if needed
        if rotation is not None:
            if rotation.shape[-1] == 4:
                rotations = rotation  # Already quaternions
            elif rotation.shape[-1] == 3:
                # Convert from axis-angle or euler
                rotations = self._axis_angle_to_quaternion(rotation)
            else:
                rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * xyz.shape[0])
        else:
            rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * xyz.shape[0])
        
        # Handle opacity
        if opacity is not None:
            if opacity.min() < 0:  # Likely logit
                opacity = torch.sigmoid(opacity)
        else:
            opacity = torch.ones(xyz.shape[0], 1) * 0.5
        
        # Combine features
        if features_dc is not None:
            if features_rest is not None:
                # Ensure features_dc has shape (N, 3) -> (N, 3, 1)
                if len(features_dc.shape) == 2:
                    features_dc = features_dc.unsqueeze(2)
                # Ensure features_rest has correct shape
                if len(features_rest.shape) == 3:
                    features = torch.cat([features_dc, features_rest], dim=2)
                else:
                    features = features_dc
            else:
                if len(features_dc.shape) == 2:
                    features = features_dc.unsqueeze(2)
                else:
                    features = features_dc
        else:
            # Default to gray
            features = torch.ones(xyz.shape[0], 3, 1) * 0.5
        
        # Extract time information and deformations
        t = torch.zeros(xyz.shape[0], 1)
        vel = torch.zeros_like(xyz)
        
        if deformation is not None:
            # Try to extract velocity or deformation field
            if isinstance(deformation, dict):
                if 'velocity' in deformation:
                    vel = deformation['velocity']
                elif 'deformation_field' in deformation:
                    # Approximate as velocity
                    deform_field = deformation['deformation_field']
                    if len(deform_field.shape) == 4:  # (T, N, 3) or (N, T, 3)
                        # Take mean deformation as velocity approximation
                        vel = deform_field.mean(dim=0 if deform_field.shape[0] < deform_field.shape[1] else 1)
                    else:
                        vel = deform_field
                
                if 'timestamps' in deformation:
                    timestamps = deformation['timestamps']
                    if len(timestamps) > 0:
                        # Assign time centers based on timestamps
                        t = torch.full((xyz.shape[0], 1), timestamps.mean().item())
        
        # Prepare output data
        data = {
            'xyz': xyz.float(),
            'vel': vel.float(),
            't': t.float(),
            'scales': scales.float(),
            'scale_t': torch.ones(xyz.shape[0], 1) * 0.1,  # Default temporal scale
            'rotations': rotations.float(),
            'opacity': opacity.float(),
            'features': features.float(),
            'metadata': {
                'source_format': 'Fudan4DGS',
                'has_deformation': deformation is not None,
                'num_points': xyz.shape[0]
            }
        }
        
        # Store deformation network if present (for advanced usage)
        if deformation is not None and isinstance(deformation, dict):
            data['metadata']['deformation_info'] = {
                'type': deformation.get('type', 'unknown'),
                'params': deformation.get('num_params', 0)
            }
        
        return data
    
    def save(self, data: Dict[str, Any], path: str) -> None:
        """Save data in Fudan 4DGS format."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Prepare Gaussian parameters
        gaussian_params = {
            '_xyz': data['xyz'].cpu(),
            '_scaling': torch.log(data['scales'].cpu()),  # Convert to log scale
            '_rotation': data['rotations'].cpu(),
            '_opacity': torch.logit(data['opacity'].cpu().clamp(0.01, 0.99)),  # Convert to logit
        }
        
        # Handle features
        features = data['features'].cpu()
        if features.shape[2] > 0:
            gaussian_params['_features_dc'] = features[..., 0]
            if features.shape[2] > 1:
                # features_rest shape should be (N, 3, K-1) where K is number of SH coeffs
                gaussian_params['_features_rest'] = features[..., 1:]
        
        # Prepare deformation data if velocity is non-zero
        vel = data.get('vel', None)
        if vel is not None and vel.abs().max() > 1e-6:
            deformation = {
                'velocity': vel.cpu(),
                'timestamps': data.get('t', torch.zeros(1)).cpu(),
                'type': 'velocity_field'
            }
        else:
            deformation = None
        
        # Create checkpoint
        checkpoint = {
            'gaussians': gaussian_params,
            'iteration': data.get('metadata', {}).get('iteration', 0),
        }
        
        if deformation is not None:
            checkpoint['deformation'] = deformation
        
        # Add metadata
        checkpoint['metadata'] = {
            'format': 'Fudan4DGS',
            'version': '1.0',
            'num_points': data['xyz'].shape[0],
            'has_deformation': deformation is not None
        }
        
        # Save checkpoint
        if path.endswith('.pth') or path.endswith('.pt'):
            torch.save(checkpoint, path)
        else:
            # Save to directory
            os.makedirs(path, exist_ok=True)
            torch.save(checkpoint, os.path.join(path, 'checkpoint.pth'))
            
            # Also save metadata as JSON for easy inspection
            metadata_path = os.path.join(path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json_metadata = {
                    k: v.tolist() if torch.is_tensor(v) else v
                    for k, v in checkpoint['metadata'].items()
                }
                json.dump(json_metadata, f, indent=2)
    
    def _extract_gaussian_params(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract Gaussian parameters from model state dict."""
        gaussian_params = {}
        
        # Common prefixes in Fudan's implementation
        prefixes = ['gaussians.', 'model.gaussians.', '']
        
        for prefix in prefixes:
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    clean_key = key[len(prefix):]
                    if any(param in clean_key for param in ['xyz', 'scaling', 'rotation', 'opacity', 'features']):
                        gaussian_params[clean_key] = value
        
        return gaussian_params
    
    def _axis_angle_to_quaternion(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle representation to quaternions."""
        angle = axis_angle.norm(dim=-1, keepdim=True)
        axis = axis_angle / (angle + 1e-8)
        
        half_angle = angle * 0.5
        sin_half = torch.sin(half_angle)
        cos_half = torch.cos(half_angle)
        
        quaternions = torch.cat([
            cos_half,
            axis * sin_half
        ], dim=-1)
        
        return self.normalize_quaternion(quaternions)
    
    def load_sequence(self, path: str, timestamps: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Load a sequence of Fudan 4DGS checkpoints.
        
        Args:
            path: Directory containing sequence of checkpoints
            timestamps: Optional list of timestamps for each frame
            
        Returns:
            List of data dictionaries, one per timestamp
        """
        sequence_data = []
        
        # Find all checkpoint files
        checkpoint_files = sorted(Path(path).glob('*.pth'))
        if not checkpoint_files:
            checkpoint_files = sorted(Path(path).glob('*.pt'))
        
        if not checkpoint_files:
            # Try numbered directories
            dirs = sorted([d for d in Path(path).iterdir() if d.is_dir()])
            checkpoint_files = [d / 'checkpoint.pth' for d in dirs]
        
        # Load each checkpoint
        for i, ckpt_path in enumerate(checkpoint_files):
            if ckpt_path.exists():
                data = self.load(str(ckpt_path))
                
                # Set timestamp
                if timestamps and i < len(timestamps):
                    data['t'] = torch.full_like(data['t'], timestamps[i])
                else:
                    data['t'] = torch.full_like(data['t'], float(i))
                
                sequence_data.append(data)
        
        return sequence_data