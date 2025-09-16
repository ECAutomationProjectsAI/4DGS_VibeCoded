import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Spherical Harmonics utilities (degree up to 3)
# Based on common 3DGS usage. We implement a small subset for RGB shading.

def num_sh_coeffs(degree: int):
    return (degree + 1) ** 2


def sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """
    Compute real SH basis up to degree for unit directions.
    dirs: [N, 3] normalized viewing directions
    returns: [N, (degree+1)^2]
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    result = []
    # l=0
    result.append(torch.full_like(x, 0.2820947918))  # Y00
    if degree >= 1:
        result += [
            -0.4886025119 * y,  # Y1-1
            0.4886025119 * z,   # Y10
            -0.4886025119 * x,  # Y11
        ]
    if degree >= 2:
        result += [
            1.0925484306 * x * y,                     # Y2-2
            -1.0925484306 * y * z,                    # Y2-1
            0.3153915652 * (3.0 * z * z - 1.0),       # Y20
            -1.0925484306 * x * z,                    # Y21
            0.5462742153 * (x * x - y * y),           # Y22
        ]
    if degree >= 3:
        result += [
            -0.5900435899 * y * (3 * x * x - y * y),                 # Y3-3
            2.8906114426 * x * y * z,                                # Y3-2
            -0.4570457995 * y * (5 * z * z - 1.0),                   # Y3-1
            0.3731763326 * z * (5 * z * z - 3.0),                    # Y30
            -0.4570457995 * x * (5 * z * z - 1.0),                   # Y31
            1.4453057213 * z * (x * x - y * y),                      # Y32
            -0.5900435899 * x * (x * x - 3 * y * y),                 # Y33
        ]
    return torch.stack(result, dim=1)


def shade_sh(rgb_sh: torch.Tensor, view_dir: torch.Tensor, degree: int) -> torch.Tensor:
    """
    rgb_sh: [N, 3, C] where C=(degree+1)^2
    view_dir: [N, 3] unit vectors
    return: [N, 3]
    """
    B = sh_basis(degree, view_dir)  # [N, C]
    return torch.einsum('n c, n k c -> n k', B, rgb_sh)
