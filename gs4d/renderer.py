import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sh import shade_sh


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: [N,4] with (w,x,y,z). Return [N,3,3]
    """
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    N = q.shape[0]
    R = torch.empty(N, 3, 3, device=q.device, dtype=q.dtype)
    R[:,0,0] = 1 - 2*(y*y + z*z)
    R[:,0,1] = 2*(x*y - z*w)
    R[:,0,2] = 2*(x*z + y*w)
    R[:,1,0] = 2*(x*y + z*w)
    R[:,1,1] = 1 - 2*(x*x + z*z)
    R[:,1,2] = 2*(y*z - x*w)
    R[:,2,0] = 2*(x*z - y*w)
    R[:,2,1] = 2*(y*z + x*w)
    R[:,2,2] = 1 - 2*(x*x + y*y)
    return R


def project_points(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, xyz: torch.Tensor):
    """
    K: [3,3], R: [3,3], t: [3], xyz: [N,3]
    return: xy [N,2], z [N]
    """
    Xc = (R @ xyz.T).T + t[None]
    z = Xc[:, 2].clamp(min=1e-6)
    x = Xc[:, 0] / z
    y = Xc[:, 1] / z
    xy = torch.stack([x, y], dim=-1)
    uv = (K @ torch.stack([x, y, torch.ones_like(x)], dim=0)).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv, z, Xc


def forward_splat(
    xyz: torch.Tensor,
    tvals: torch.Tensor,
    scales: torch.Tensor,
    scale_t: torch.Tensor,
    opacity: torch.Tensor,
    rgb_sh: torch.Tensor,
    view_dir: torch.Tensor,
    sh_degree: int,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    H: int,
    W: int,
    time: float,
    max_points_per_pixel: int = 32,
    quat: torch.Tensor = None,
    elliptical: bool = True
):
    """
    A simple CPU-friendly differentiable splatting and alpha compositing.
    This is a naive O(N*H*W) approach, meant for validation and small scenes.

    Returns: image [3,H,W], alpha [1,H,W], depth [1,H,W]
    """
    device = xyz.device
    N = xyz.shape[0]

    # Temporal weight based on distance in time (temporal Gaussian support)
    dt = (tvals.squeeze(-1) - time) / (scale_t.squeeze(-1) + 1e-6)
    w_t = torch.exp(-0.5 * dt * dt)  # [N]

    uv, z, Xc = project_points(K, R, t, xyz)

    # Rough 2D footprint size from scale and depth
    sx = scales[:, 0]
    sy = scales[:, 1]
    # approximate projected radius in pixels
    fx = K[0, 0]
    fy = K[1, 1]
    r_pix = 0.5 * (fx * sx / z + fy * sy / z)  # [N]

    # Prepare full-frame coordinate grid
    ys = torch.arange(0, H, device=device).float()
    xs = torch.arange(0, W, device=device).float()
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]

    # Init outputs
    image = torch.zeros(3, H, W, device=device)
    alpha = torch.zeros(1, H, W, device=device)
    depth = torch.zeros(1, H, W, device=device)

    # Sort points by depth (back to front for over)
    order = torch.argsort(z, descending=True)

    # Precompute shaded colors
    rgb = shade_sh(rgb_sh, view_dir, sh_degree)  # [N,3]
    rgb = rgb.clamp(0.0, 1.0)

    # Precompute per-point 3D covariance orientation
    if elliptical and quat is not None:
        Rpt = quat_to_rotmat(quat)  # [N,3,3]
        S = torch.zeros(N,3,3, device=device, dtype=xyz.dtype)
        S[:,0,0] = scales[:,0]**2
        S[:,1,1] = scales[:,1]**2
        S[:,2,2] = scales[:,2]**2
        Sigma3D = Rpt @ S @ Rpt.transpose(1,2)  # [N,3,3]

    for idx in order.tolist():
        u, v = uv[idx]
        if z[idx] <= 0:
            continue
        a = torch.clamp(opacity[idx], 0.0, 1.0) * w_t[idx]
        if not elliptical or quat is None:
            r = torch.clamp(r_pix[idx], min=1.0, max=20.0)
            du = grid_x - u
            dv = grid_y - v
            s2 = (r * 0.5) ** 2
            w = torch.exp(-0.5 * (du * du + dv * dv) / (s2 + 1e-8)) * a  # [H,W]
            one_minus_A = (1.0 - alpha)
            image = image + (one_minus_A * w[None]) * rgb[idx][:, None, None]
            alpha_new = alpha + one_minus_A * w[None]
            depth = (depth * alpha + z[idx] * (one_minus_A * w[None])) / (alpha_new + 1e-8)
            alpha = alpha_new
        else:
            # Elliptical footprint via first-order projection
            # Jacobian J at point
            Xc = Xc = (R @ xyz[idx]).view(3) + t
            zx = Xc[2].clamp(min=1e-6)
            x = Xc[0]; y = Xc[1]
            fx = K[0,0]; fy = K[1,1]
            J = torch.tensor([[fx/zx, 0.0, -fx*x/(zx*zx)],
                              [0.0, fy/zx, -fy*y/(zx*zx)]], device=device, dtype=xyz.dtype)
            # Image-space covariance, symmetrize and regularize
            Sigma_img = J @ Sigma3D[idx] @ J.transpose(0,1)
            Sigma_img = 0.5 * (Sigma_img + Sigma_img.transpose(0,1))
            Sigma_img = Sigma_img + 1e-8*torch.eye(2, device=device, dtype=xyz.dtype)
            # Determine bbox via eigenvalues (robust)
            try:
                eigvals, _ = torch.linalg.eigh(Sigma_img)
                sigma_max = torch.sqrt(torch.clamp(eigvals.max(), min=1e-12))
            except RuntimeError:
                # Fallback: use diagonal upper bound if eigh fails
                diag = torch.diagonal(Sigma_img, 0)
                sigma_max = torch.sqrt(torch.clamp(diag.max(), min=1e-12))
            r = torch.clamp(3.0 * sigma_max, min=1.0, max=40.0)  # 3-sigma box
            u0 = int(torch.floor(u - r).item()); v0 = int(torch.floor(v - r).item())
            u1 = int(torch.ceil(u + r).item()); v1 = int(torch.ceil(v + r).item())
            if u1 < 0 or v1 < 0 or u0 >= W or v0 >= H:
                continue
            uu0 = max(0, u0); vv0 = max(0, v0); uu1 = min(W - 1, u1); vv1 = min(H - 1, v1)
            ys_loc = torch.arange(vv0, vv1 + 1, device=device).float()
            xs_loc = torch.arange(uu0, uu1 + 1, device=device).float()
            gy, gx = torch.meshgrid(ys_loc, xs_loc, indexing='ij')
            d = torch.stack([gx - u, gy - v], dim=-1)  # [h,w,2]
            # inv Sigma_img (robust determinant)
            a11 = Sigma_img[0,0]; a12 = Sigma_img[0,1]; a22 = Sigma_img[1,1]
            det = a11*a22 - a12*a12
            det = torch.where(det.abs() < 1e-12, det.sign()*1e-12 + 1e-12, det)
            i11 = a22/det; i22 = a11/det; i12 = -a12/det
            # mahalanobis distance
            dx = d[...,0]; dy = d[...,1]
            md2 = i11*dx*dx + 2*i12*dx*dy + i22*dy*dy
            w = torch.exp(-0.5 * md2) * a  # [h,w]
            # Compose
            patch_A = alpha[0, vv0:vv1+1, uu0:uu1+1]
            one_minus_A = (1.0 - patch_A)
            new_img_patch = image[:, vv0:vv1+1, uu0:uu1+1] + (one_minus_A * w)[None] * rgb[idx][:, None, None]
            new_A_patch = patch_A + one_minus_A * w
            new_D_patch = (depth[0, vv0:vv1+1, uu0:uu1+1] * patch_A + z[idx] * (one_minus_A * w)) / (new_A_patch + 1e-8)
            image[:, vv0:vv1+1, uu0:uu1+1] = new_img_patch
            alpha[0, vv0:vv1+1, uu0:uu1+1] = new_A_patch
            depth[0, vv0:vv1+1, uu0:uu1+1] = new_D_patch

    return image.clamp(0, 1), alpha.clamp(0, 1), depth
