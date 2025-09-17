"""Temporal consistency losses for 4D Gaussian Splatting.

These losses ensure smooth and consistent motion across time:
1. Velocity smoothness - penalizes sudden velocity changes
2. Position consistency - ensures smooth trajectories
3. Appearance consistency - maintains color/opacity stability
4. Deformation regularization - prevents excessive distortions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class TemporalConsistencyLoss(nn.Module):
    """Combined temporal consistency loss for 4DGS."""
    
    def __init__(
        self,
        velocity_smooth_weight: float = 1.0,
        position_smooth_weight: float = 0.5,
        appearance_weight: float = 0.2,
        deformation_reg_weight: float = 0.1,
        rigidity_weight: float = 0.05
    ):
        """
        Initialize temporal consistency loss.
        
        Args:
            velocity_smooth_weight: Weight for velocity smoothness
            position_smooth_weight: Weight for position trajectory smoothness
            appearance_weight: Weight for appearance consistency
            deformation_reg_weight: Weight for deformation regularization
            rigidity_weight: Weight for local rigidity constraint
        """
        super().__init__()
        self.velocity_smooth_weight = velocity_smooth_weight
        self.position_smooth_weight = position_smooth_weight
        self.appearance_weight = appearance_weight
        self.deformation_reg_weight = deformation_reg_weight
        self.rigidity_weight = rigidity_weight
    
    def forward(
        self,
        positions_t0: torch.Tensor,
        positions_t1: torch.Tensor,
        velocities: torch.Tensor,
        colors_t0: torch.Tensor,
        colors_t1: torch.Tensor,
        opacities_t0: torch.Tensor,
        opacities_t1: torch.Tensor,
        scales_t0: Optional[torch.Tensor] = None,
        scales_t1: Optional[torch.Tensor] = None,
        dt: float = 1.0,
        neighbor_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute temporal consistency losses.
        
        Args:
            positions_t0: Positions at time t (N, 3)
            positions_t1: Positions at time t+dt (N, 3)
            velocities: Velocities (N, 3)
            colors_t0: Colors at time t (N, 3)
            colors_t1: Colors at time t+dt (N, 3)
            opacities_t0: Opacities at time t (N, 1)
            opacities_t1: Opacities at time t+dt (N, 1)
            scales_t0: Optional scales at time t (N, 3)
            scales_t1: Optional scales at time t+dt (N, 3)
            dt: Time step
            neighbor_indices: Optional neighbor indices for rigidity (N, K)
            
        Returns:
            Dictionary of individual loss components
        """
        losses = {}
        
        # 1. Velocity smoothness loss
        if self.velocity_smooth_weight > 0:
            losses['velocity_smooth'] = self._velocity_smoothness_loss(
                positions_t0, positions_t1, velocities, dt
            ) * self.velocity_smooth_weight
        
        # 2. Position trajectory smoothness
        if self.position_smooth_weight > 0:
            losses['position_smooth'] = self._position_smoothness_loss(
                positions_t0, positions_t1, velocities, dt
            ) * self.position_smooth_weight
        
        # 3. Appearance consistency
        if self.appearance_weight > 0:
            losses['appearance'] = self._appearance_consistency_loss(
                colors_t0, colors_t1, opacities_t0, opacities_t1
            ) * self.appearance_weight
        
        # 4. Deformation regularization
        if self.deformation_reg_weight > 0 and scales_t0 is not None and scales_t1 is not None:
            losses['deformation'] = self._deformation_regularization_loss(
                scales_t0, scales_t1
            ) * self.deformation_reg_weight
        
        # 5. Local rigidity constraint
        if self.rigidity_weight > 0 and neighbor_indices is not None:
            losses['rigidity'] = self._local_rigidity_loss(
                positions_t0, positions_t1, neighbor_indices
            ) * self.rigidity_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def _velocity_smoothness_loss(
        self,
        positions_t0: torch.Tensor,
        positions_t1: torch.Tensor,
        velocities: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Penalize deviation from constant velocity model.
        
        The predicted position should match: x(t+dt) = x(t) + v * dt
        """
        predicted_positions = positions_t0 + velocities * dt
        velocity_error = (predicted_positions - positions_t1).norm(dim=-1)
        return velocity_error.mean()
    
    def _position_smoothness_loss(
        self,
        positions_t0: torch.Tensor,
        positions_t1: torch.Tensor,
        velocities: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Encourage smooth position trajectories using acceleration penalty.
        
        Computes finite difference approximation of acceleration.
        """
        # Estimate velocity from positions
        empirical_velocity = (positions_t1 - positions_t0) / dt
        
        # Acceleration is change in velocity
        acceleration = (empirical_velocity - velocities) / dt
        
        # Penalize large accelerations
        return acceleration.norm(dim=-1).mean()
    
    def _appearance_consistency_loss(
        self,
        colors_t0: torch.Tensor,
        colors_t1: torch.Tensor,
        opacities_t0: torch.Tensor,
        opacities_t1: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage consistent appearance over time.
        
        Penalizes rapid changes in color and opacity.
        """
        color_diff = (colors_t1 - colors_t0).norm(dim=-1)
        opacity_diff = (opacities_t1 - opacities_t0).abs().squeeze(-1)
        
        # Combined appearance change
        appearance_loss = color_diff.mean() + opacity_diff.mean()
        
        return appearance_loss
    
    def _deformation_regularization_loss(
        self,
        scales_t0: torch.Tensor,
        scales_t1: torch.Tensor
    ) -> torch.Tensor:
        """
        Regularize scale changes to prevent excessive deformation.
        """
        scale_ratio = scales_t1 / (scales_t0 + 1e-6)
        
        # Penalize scale ratios far from 1
        deformation_loss = (scale_ratio - 1.0).abs().mean()
        
        return deformation_loss
    
    def _local_rigidity_loss(
        self,
        positions_t0: torch.Tensor,
        positions_t1: torch.Tensor,
        neighbor_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage local rigidity by preserving distances to neighbors.
        
        Args:
            positions_t0: Positions at time t (N, 3)
            positions_t1: Positions at time t+1 (N, 3)
            neighbor_indices: Indices of K nearest neighbors (N, K)
        """
        N, K = neighbor_indices.shape
        
        # Get neighbor positions
        neighbor_pos_t0 = positions_t0[neighbor_indices]  # (N, K, 3)
        neighbor_pos_t1 = positions_t1[neighbor_indices]  # (N, K, 3)
        
        # Compute distances to neighbors
        dist_t0 = (positions_t0.unsqueeze(1) - neighbor_pos_t0).norm(dim=-1)  # (N, K)
        dist_t1 = (positions_t1.unsqueeze(1) - neighbor_pos_t1).norm(dim=-1)  # (N, K)
        
        # Penalize distance changes
        dist_change = (dist_t1 - dist_t0).abs()
        
        return dist_change.mean()


class MotionSegmentationLoss(nn.Module):
    """Loss for encouraging motion segmentation/grouping."""
    
    def __init__(self, num_segments: int = 5, temperature: float = 1.0):
        """
        Initialize motion segmentation loss.
        
        Args:
            num_segments: Expected number of motion segments
            temperature: Temperature for soft assignment
        """
        super().__init__()
        self.num_segments = num_segments
        self.temperature = temperature
    
    def forward(
        self,
        velocities: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage velocities to cluster into distinct motion patterns.
        
        Args:
            velocities: Point velocities (N, 3)
            positions: Point positions (N, 3)
            
        Returns:
            Segmentation loss
        """
        N = velocities.shape[0]
        
        # Compute velocity similarity matrix
        vel_norm = velocities / (velocities.norm(dim=-1, keepdim=True) + 1e-6)
        similarity = torch.mm(vel_norm, vel_norm.t())  # (N, N)
        
        # Soft clustering using similarity
        affinity = torch.exp(similarity / self.temperature)
        
        # Normalize to get soft assignments
        assignments = affinity / affinity.sum(dim=1, keepdim=True)
        
        # Entropy regularization to encourage distinct clusters
        entropy = -(assignments * torch.log(assignments + 1e-8)).sum(dim=1)
        
        # Encourage low entropy (distinct clusters)
        return entropy.mean()


class CyclicConsistencyLoss(nn.Module):
    """Loss for cyclic/periodic motion consistency."""
    
    def __init__(self, period: float = 1.0):
        """
        Initialize cyclic consistency loss.
        
        Args:
            period: Expected period of cyclic motion
        """
        super().__init__()
        self.period = period
    
    def forward(
        self,
        positions_t0: torch.Tensor,
        positions_tT: torch.Tensor,
        is_cyclic_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encourage positions to return to start after one period.
        
        Args:
            positions_t0: Initial positions (N, 3)
            positions_tT: Positions after one period (N, 3)
            is_cyclic_mask: Optional mask for cyclic points (N,)
            
        Returns:
            Cyclic consistency loss
        """
        position_diff = (positions_tT - positions_t0).norm(dim=-1)
        
        if is_cyclic_mask is not None:
            position_diff = position_diff * is_cyclic_mask
            return position_diff.sum() / (is_cyclic_mask.sum() + 1e-6)
        
        return position_diff.mean()


def compute_knn_indices(positions: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Compute k-nearest neighbor indices for each point.
    
    Args:
        positions: Point positions (N, 3)
        k: Number of neighbors
        
    Returns:
        Neighbor indices (N, k)
    """
    N = positions.shape[0]
    max_points_for_knn = 5000  # Limit to avoid OOM
    
    # If too many points, subsample for KNN
    if N > max_points_for_knn:
        # Randomly sample subset for KNN
        sample_idx = torch.randperm(N, device=positions.device)[:max_points_for_knn]
        sampled_pos = positions[sample_idx]
        
        # Compute KNN on sampled points only
        dist_matrix = torch.cdist(sampled_pos, sampled_pos)  # (sample, sample)
        _, local_indices = torch.topk(dist_matrix, min(k + 1, sampled_pos.shape[0]), 
                                      dim=1, largest=False)
        
        # Map back to global indices
        neighbor_indices = sample_idx[local_indices[:, 1:]]  # Skip self
        
        # For all points, create neighbor list (sampled points get real neighbors)
        all_neighbors = torch.zeros(N, k, dtype=torch.long, device=positions.device)
        all_neighbors[sample_idx] = neighbor_indices
        
        # Unsampled points get random neighbors (approximation)
        unsampled = torch.ones(N, dtype=torch.bool, device=positions.device)
        unsampled[sample_idx] = False
        if unsampled.any():
            # Each unsampled point gets random sampled points as neighbors
            for i in unsampled.nonzero().squeeze(-1):
                rand_neighbors = sample_idx[torch.randperm(len(sample_idx))[:k]]
                all_neighbors[i] = rand_neighbors
        
        return all_neighbors
    else:
        # Original computation for small point clouds
        dist_matrix = torch.cdist(positions, positions)  # (N, N)
        _, indices = torch.topk(dist_matrix, min(k + 1, N), dim=1, largest=False)
        neighbor_indices = indices[:, 1:min(k+1, indices.shape[1])]  # Skip self
        
        # Pad if needed
        if neighbor_indices.shape[1] < k:
            pad_size = k - neighbor_indices.shape[1]
            padding = torch.zeros(N, pad_size, dtype=torch.long, device=positions.device)
            neighbor_indices = torch.cat([neighbor_indices, padding], dim=1)
        
        return neighbor_indices


def temporal_consistency_regularizer(
    model,
    timestamps: torch.Tensor,
    base_weight: float = 0.01,
    use_knn: bool = True,
    k_neighbors: int = 10
) -> torch.Tensor:
    """
    Compute temporal consistency regularization for the entire model.
    
    Args:
        model: GaussianModel4D instance
        timestamps: List of timestamps to evaluate (T,)
        base_weight: Base weight for regularization
        use_knn: Whether to use k-nearest neighbors for rigidity
        k_neighbors: Number of neighbors for rigidity constraint
        
    Returns:
        Total regularization loss
    """
    loss_fn = TemporalConsistencyLoss()
    total_loss = 0.0
    
    # Get base positions and velocities
    base_positions = model.xyz
    velocities = model.vel
    
    # Compute neighbors if needed
    neighbor_indices = None
    if use_knn:
        neighbor_indices = compute_knn_indices(base_positions, k_neighbors)
    
    # Evaluate at consecutive timestamps
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        dt = (t1 - t0).item()
        
        # Get positions at both times
        pos_t0 = model.position_at_time(t0.item())
        pos_t1 = model.position_at_time(t1.item())
        
        # Get colors (from SH coefficients)
        colors_t0 = model.rgb_sh[:, :, 0]  # DC component
        colors_t1 = colors_t0  # Assuming color doesn't change for now
        
        # Get opacities
        opacities_t0 = model.opacity
        opacities_t1 = opacities_t0  # Assuming opacity doesn't change
        
        # Compute losses
        losses = loss_fn(
            positions_t0=pos_t0,
            positions_t1=pos_t1,
            velocities=velocities,
            colors_t0=colors_t0,
            colors_t1=colors_t1,
            opacities_t0=opacities_t0,
            opacities_t1=opacities_t1,
            scales_t0=model.scales,
            scales_t1=model.scales,  # Assuming scales don't change
            dt=dt,
            neighbor_indices=neighbor_indices
        )
        
        total_loss = total_loss + losses['total']
    
    # Average over time steps
    total_loss = total_loss / (len(timestamps) - 1)
    
    return total_loss * base_weight