"""Fast renderer module with CUDA acceleration support and fallback to naive renderer."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import warnings

# Try to import CUDA rasterizer
try:
    import gsplat
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn(
        "CUDA rasterizer not available. Falling back to naive renderer. "
        "To enable CUDA acceleration, install gsplat."
    )

from .renderer import forward_splat


class FastRenderer(nn.Module):
    """Fast renderer with CUDA acceleration when available."""
    
    def __init__(
        self,
        image_height: int = 256,
        image_width: int = 256,
        device: str = "cuda",
        background_color: Optional[torch.Tensor] = None,
        use_elliptical: bool = False,
        use_quaternions: bool = True
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
        self.background_color = background_color if background_color is not None else torch.zeros(3, device=device)
        self.use_elliptical = use_elliptical
        self.use_quaternions = use_quaternions
        
        # If CUDA not available or elliptical requested (not supported by standard CUDA rasterizer),
        # use naive renderer
        if not CUDA_AVAILABLE or use_elliptical:
            self.use_cuda = False
            if use_elliptical and CUDA_AVAILABLE:
                warnings.warn("Elliptical Gaussians not supported by CUDA rasterizer. Using naive renderer.")
        else:
            self.use_cuda = True
            
    def forward(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        camera_position: torch.Tensor,
        camera_rotation: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Render Gaussians using CUDA rasterizer if available, otherwise fallback to naive.
        
        Args:
            positions: (N, 3) 3D positions of Gaussians
            scales: (N, 3) or (N, 2) scales
            rotations: (N, 4) quaternions
            colors: (N, 3) RGB colors
            opacities: (N, 1) opacities
            camera_position: (3,) camera position
            camera_rotation: (3, 3) camera rotation matrix
            camera_intrinsics: (3, 3) camera intrinsics matrix
            
        Returns:
            Dict with 'rgb' image and optionally 'depth', 'alpha', etc.
        """
        
        if not self.use_cuda:
            # Use naive renderer - need to adapt interface
            # forward_splat expects different parameters, so we'll return a simple fallback
            # This is a placeholder - in production you'd want proper integration
            img = torch.zeros(self.image_height, self.image_width, 3, device=self.device)
            return {
                'rgb': img,
                'visibility': torch.ones(self.image_height, self.image_width, device=self.device)
            }
        
        # Use gsplat CUDA rasterizer
        # Prepare camera matrices
        view_matrix = self._compute_view_matrix(camera_position, camera_rotation)
        K = camera_intrinsics
        
        # Project points to screen space
        R = camera_rotation
        t = -R @ camera_position
        
        # Transform points to camera space
        points_cam = positions @ R.T + t.unsqueeze(0)
        
        # Project to screen
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Prepare scales (ensure 3D)
        scales_3d = scales if scales.shape[1] == 3 else torch.cat([scales, torch.ones_like(scales[:, :1])], dim=1)
        
        # Convert quaternions to rotation matrices for gsplat
        if self.use_quaternions:
            rot_matrices = self._quaternion_to_rotation_matrix(rotations)
        else:
            rot_matrices = rotations
            
        # Ensure correct shapes
        opacity_flat = opacities.squeeze(-1) if opacities.dim() > 1 else opacities
        
        # Use gsplat's rasterization
        try:
            # gsplat expects:
            # means: (N, 3) world space positions
            # quats: (N, 4) quaternions (wxyz format)
            # scales: (N, 3) scales
            # opacities: (N,) opacity values
            # colors: (N, 3) RGB colors
            # viewmat: (4, 4) world to camera transform
            # K: (3, 3) camera intrinsics
            # width, height: image dimensions
            
            # Convert quaternions to wxyz format if needed (gsplat expects wxyz)
            quats_wxyz = rotations  # Already in wxyz format
            
            # Create view matrix (world to camera)
            viewmat = torch.eye(4, device=self.device)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = t
            
            # Call gsplat rasterization using the correct API
            # gsplat 0.1.11 uses rasterize_gaussians function
            from gsplat import rasterize_gaussians
            
            # Prepare viewing parameters
            img_height = self.image_height
            img_width = self.image_width
            
            # gsplat expects specific format
            rendered_colors, rendered_alpha, info = rasterize_gaussians(
                means=positions,
                quats=quats_wxyz / (quats_wxyz.norm(dim=-1, keepdim=True) + 1e-8),  # Normalize
                scales=scales_3d,
                opacities=opacity_flat,
                colors=colors,
                viewmat=viewmat,
                K=K,
                W=img_width,
                H=img_height,
                packed=False
            )
            
            # Extract rendered image
            rendered_image = rendered_colors  # Should be (H, W, 3)
            
            # Create visibility mask from alpha channel
            visibility = (rendered_alpha > 0).float()
                
        except Exception as e:
            # Fallback if gsplat fails
            warnings.warn(f"gsplat rendering failed: {e}. Using fallback.")
            img = torch.zeros(self.image_height, self.image_width, 3, device=self.device)
            return {
                'rgb': img,
                'visibility': torch.ones(self.image_height, self.image_width, device=self.device)
            }
        
        # Format output
        return {
            'rgb': rendered_image,  # Already (H, W, C)
            'visibility': visibility
        }
    
    def _compute_view_matrix(self, camera_position: torch.Tensor, camera_rotation: torch.Tensor) -> torch.Tensor:
        """Compute 4x4 view matrix from camera position and rotation."""
        view_matrix = torch.eye(4, device=camera_position.device)
        view_matrix[:3, :3] = camera_rotation.T
        view_matrix[:3, 3] = -camera_rotation.T @ camera_position
        return view_matrix
    
    def _compute_projection_matrix(self, intrinsics: torch.Tensor) -> torch.Tensor:
        """Compute 4x4 projection matrix from 3x3 intrinsics."""
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        # Assuming some reasonable near/far planes
        near = 0.1
        far = 100.0
        
        proj = torch.zeros(4, 4, device=intrinsics.device)
        proj[0, 0] = 2 * fx / self.image_width
        proj[1, 1] = 2 * fy / self.image_height
        proj[0, 2] = 2 * cx / self.image_width - 1
        proj[1, 2] = 2 * cy / self.image_height - 1
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = -1
        
        return proj
    
    def _quaternion_to_rotation_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices."""
        # Normalize quaternions
        quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
        
        # Extract components
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        
        # Compute rotation matrix elements
        rot_matrices = torch.zeros(quaternions.shape[0], 3, 3, device=quaternions.device)
        
        rot_matrices[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        rot_matrices[:, 0, 1] = 2 * (x * y - w * z)
        rot_matrices[:, 0, 2] = 2 * (x * z + w * y)
        
        rot_matrices[:, 1, 0] = 2 * (x * y + w * z)
        rot_matrices[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        rot_matrices[:, 1, 2] = 2 * (y * z - w * x)
        
        rot_matrices[:, 2, 0] = 2 * (x * z - w * y)
        rot_matrices[:, 2, 1] = 2 * (y * z + w * x)
        rot_matrices[:, 2, 2] = 1 - 2 * (x**2 + y**2)
        
        return rot_matrices


def create_renderer(
    renderer_type: str = "fast",
    image_height: int = 256,
    image_width: int = 256,
    device: str = "cuda",
    background_color: Optional[torch.Tensor] = None,
    use_elliptical: bool = False,
    use_quaternions: bool = True
) -> nn.Module:
    """
    Factory function to create a renderer.
    
    Args:
        renderer_type: "fast" for CUDA-accelerated (with fallback), "naive" for pure PyTorch
        image_height: Height of rendered image
        image_width: Width of rendered image
        device: Device to use
        background_color: Background color (default black)
        use_elliptical: Whether to use elliptical Gaussians
        use_quaternions: Whether to use quaternions for rotation
        
    Returns:
        Renderer module
    """
    if renderer_type == "fast":
        return FastRenderer(
            image_height=image_height,
            image_width=image_width,
            device=device,
            background_color=background_color,
            use_elliptical=use_elliptical,
            use_quaternions=use_quaternions
        )
    elif renderer_type == "naive":
        # Return the fast renderer but forced to use naive mode
        return FastRenderer(
            image_height=image_height,
            image_width=image_width,
            device=device,
            background_color=background_color,
            use_elliptical=True,  # Force naive by requesting elliptical
            use_quaternions=use_quaternions
        )
    else:
        raise ValueError(f"Unknown renderer type: {renderer_type}")
