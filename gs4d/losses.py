import math
import torch
import torch.nn.functional as F


def l1_loss(a: torch.Tensor, b: torch.Tensor):
    return (a - b).abs().mean()


def mse_loss(a: torch.Tensor, b: torch.Tensor):
    return ((a - b) ** 2).mean()


def psnr(a: torch.Tensor, b: torch.Tensor):
    mse = mse_loss(a, b).clamp(min=1e-8)
    return -10.0 * torch.log10(mse)


def _gaussian_window(window_size: int, sigma: float, device: torch.device):
    gauss = torch.tensor([math.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)], device=device)
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d @ window_1d.T
    return window_2d


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5, C1: float = 0.01**2, C2: float = 0.03**2):
    """
    img1, img2: [3,H,W] in [0,1]
    returns scalar SSIM in [0,1]
    """
    device = img1.device
    window = _gaussian_window(window_size, sigma, device).unsqueeze(0).unsqueeze(0)
    # per-channel conv
    mu1 = F.conv2d(img1.unsqueeze(0), window.expand(3,1,window_size,window_size), padding=window_size//2, groups=3)
    mu2 = F.conv2d(img2.unsqueeze(0), window.expand(3,1,window_size,window_size), padding=window_size//2, groups=3)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d((img1.unsqueeze(0))**2, window.expand(3,1,window_size,window_size), padding=window_size//2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d((img2.unsqueeze(0))**2, window.expand(3,1,window_size,window_size), padding=window_size//2, groups=3) - mu2_sq
    sigma12 = F.conv2d((img1.unsqueeze(0) * img2.unsqueeze(0)), window.expand(3,1,window_size,window_size), padding=window_size//2, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
    return ssim_map.mean()
