import math
import torch
import torch.nn as nn

from .sh import num_sh_coeffs, shade_sh

class Gaussian4D(nn.Module):
    """
    4D Gaussian primitive with parameters:
      - mean: [x, y, z]
      - velocity: [vx, vy, vz] for time-dependent mean x(t) = x + v * t
      - scale: 3D spatial log-scales (sx, sy, sz) and temporal log-scale st controlling temporal support
      - rotation: 3D rotation (quaternion) optionally extended to affect time via small omega
      - opacity: alpha in (0,1)
      - SH colors: degree D, coeffs per channel C=(D+1)^2
    """
    def __init__(self, sh_degree: int = 3, enable_time_rotation: bool = False):
        super().__init__()
        self.sh_degree = sh_degree
        self.C = num_sh_coeffs(sh_degree)
        self.enable_time_rotation = enable_time_rotation

        # Buffers filled by parent model
        self._xyz = None          # [N, 3]
        self._vel = None          # [N, 3]
        self._t = None            # [N, 1] center time (unused by simple renderer but stored)
        self._log_scale = None    # [N, 3]
        self._log_scale_t = None  # [N, 1]
        self._quat = None         # [N, 4] normalized
        self._omega_t = None      # [N, 1] optional small coupling
        self._opacity = None      # [N, 1]
        self._rgb_sh = None       # [N, 3, C]

    def n_points(self):
        return 0 if self._xyz is None else self._xyz.shape[0]

    @staticmethod
    def normalize_quat(q: torch.Tensor) -> torch.Tensor:
        return q / (q.norm(dim=-1, keepdim=True) + 1e-8)

    def parameters_dict(self):
        return {
            'xyz': self._xyz,
            'vel': self._vel,
            't': self._t,
            'log_scale': self._log_scale,
            'log_scale_t': self._log_scale_t,
            'quat': self._quat,
            'omega_t': self._omega_t,
            'opacity': self._opacity,
            'rgb_sh': self._rgb_sh,
        }

class GaussianModel4D(nn.Module):
    def __init__(self, sh_degree: int = 3, enable_time_rotation: bool = False, device='cuda'):
        super().__init__()
        self.primitive = Gaussian4D(sh_degree, enable_time_rotation)
        self.device = device

    def init_from_pcd(self, xyz: torch.Tensor, rgb: torch.Tensor, t: torch.Tensor):
        """
        xyz: [N,3], rgb: [N,3] in [0,1], t: [N,1] in [-0.5,0.5]
        Initialize minimal SH as DC, isotropic scales, small opacity, zero velocity.
        """
        N = xyz.shape[0]
        C = self.primitive.C
        self.primitive._xyz = nn.Parameter(xyz.to(self.device))
        self.primitive._vel = nn.Parameter(torch.zeros(N,3, device=self.device))
        self.primitive._t = nn.Parameter(t.to(self.device))
        self.primitive._log_scale = nn.Parameter(torch.full((N,3), -2.0, device=self.device))
        self.primitive._log_scale_t = nn.Parameter(torch.full((N,1), -2.5, device=self.device))
        quat = torch.zeros(N,4, device=self.device)
        quat[:,0] = 1.0
        self.primitive._quat = nn.Parameter(quat)
        self.primitive._omega_t = nn.Parameter(torch.zeros(N,1, device=self.device))
        self.primitive._opacity = nn.Parameter(torch.full((N,1), 0.5, device=self.device))
        rgb_sh = torch.zeros(N,3,C, device=self.device)
        rgb_sh[:,:,0] = rgb.to(self.device)
        self.primitive._rgb_sh = nn.Parameter(rgb_sh)

    @property
    def xyz(self):
        return self.primitive._xyz

    @property
    def vel(self):
        return self.primitive._vel

    @property
    def t(self):
        return self.primitive._t

    @property
    def opacity(self):
        return self.primitive._opacity.sigmoid()

    @property
    def scales(self):
        return self.primitive._log_scale.exp()

    @property
    def scale_t(self):
        return self.primitive._log_scale_t.exp()

    @property
    def quat(self):
        return Gaussian4D.normalize_quat(self.primitive._quat)

    @property
    def rgb_sh(self):
        return self.primitive._rgb_sh

    def position_at_time(self, time_scalar: float) -> torch.Tensor:
        """Compute per-point position at scalar time t: x(t) = x + v * t."""
        return self.xyz + self.vel * time_scalar
    
    def get_rgb(self, view_dirs: torch.Tensor, sh_degree: int) -> torch.Tensor:
        """Compute RGB colors from SH coefficients and viewing directions.
        
        Args:
            view_dirs: [N, 3] normalized viewing directions
            sh_degree: Active SH degree to use (0 to self.primitive.sh_degree)
            
        Returns:
            [N, 3] RGB colors
        """
        return shade_sh(self.rgb_sh, view_dirs, sh_degree)

    def prune(self, mask_keep: torch.Tensor):
        def f(x):
            return nn.Parameter(x[mask_keep]) if x is not None else None
        p = self.primitive
        p._xyz = f(p._xyz)
        p._vel = f(p._vel)
        p._t = f(p._t)
        p._log_scale = f(p._log_scale)
        p._log_scale_t = f(p._log_scale_t)
        p._quat = f(p._quat)
        p._omega_t = f(p._omega_t)
        p._opacity = f(p._opacity)
        p._rgb_sh = f(p._rgb_sh)

    def densify_clone(self, grad_mask: torch.Tensor):
        """Clone points with high gradient into two copies with slightly perturbed means."""
        if grad_mask.sum() == 0:
            return
        p = self.primitive
        idx = torch.nonzero(grad_mask, as_tuple=False).squeeze(1)
        jitter = 0.01
        def cat(a,b):
            return nn.Parameter(torch.cat([a, b], dim=0))
        p._xyz = cat(p._xyz, p._xyz[idx] + torch.randn_like(p._xyz[idx]) * jitter)
        p._vel = cat(p._vel, p._vel[idx] + torch.randn_like(p._vel[idx]) * (jitter*0.1))
        p._t = cat(p._t, p._t[idx] + torch.randn_like(p._t[idx]) * (jitter*0.1))
        p._log_scale = cat(p._log_scale, p._log_scale[idx])
        p._log_scale_t = cat(p._log_scale_t, p._log_scale_t[idx])
        p._quat = cat(p._quat, p._quat[idx])
        p._omega_t = cat(p._omega_t, p._omega_t[idx])
        p._opacity = cat(p._opacity, p._opacity[idx])
        p._rgb_sh = cat(p._rgb_sh, p._rgb_sh[idx])

    def state_dict_compact(self):
        return {k: v.detach().cpu() for k, v in self.primitive.parameters_dict().items()}

    def load_state_dict_compact(self, state):
        p = self.primitive
        p._xyz = nn.Parameter(state['xyz'].to(self.device))
        p._vel = nn.Parameter(state.get('vel', torch.zeros_like(state['xyz'])).to(self.device))
        p._t = nn.Parameter(state['t'].to(self.device))
        p._log_scale = nn.Parameter(state['log_scale'].to(self.device))
        p._log_scale_t = nn.Parameter(state['log_scale_t'].to(self.device))
        p._quat = nn.Parameter(state['quat'].to(self.device))
        p._omega_t = nn.Parameter(state['omega_t'].to(self.device))
        p._opacity = nn.Parameter(state['opacity'].to(self.device))
        p._rgb_sh = nn.Parameter(state['rgb_sh'].to(self.device))
