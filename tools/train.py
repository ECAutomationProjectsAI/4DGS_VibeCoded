import os
import sys
import json
import math
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import gs4d module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs4d.gaussians import GaussianModel4D
from gs4d.dataio import load_sequence
from gs4d.renderer import forward_splat
from gs4d.fast_renderer import create_renderer
from gs4d.losses import l1_loss, psnr, ssim
from gs4d.temporal_losses import temporal_consistency_regularizer, TemporalConsistencyLoss
from gs4d.utils import ensure_dir


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Train 4D Gaussian Splatting (from scratch)')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--mask_root', type=str, default=None, help='Optional mask folder; mask filenames should match image basenames')
    parser.add_argument('--out_dir', type=str, default='outputs/exp')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--validate_every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_points', type=int, default=50000)
    parser.add_argument('--renderer', type=str, default='naive', choices=['naive','fast'], help='Renderer backend')
    # Loss weights
    parser.add_argument('--w_ssim', type=float, default=0.2)
    parser.add_argument('--w_opa_mask', type=float, default=0.1)
    parser.add_argument('--w_temporal', type=float, default=0.01, help='Weight for temporal consistency')
    parser.add_argument('--temporal_window', type=int, default=3, help='Number of frames for temporal consistency')
    # Densify/Prune
    parser.add_argument('--densify_from_iter', type=int, default=200)
    parser.add_argument('--densify_until_iter', type=int, default=20000)
    parser.add_argument('--densification_interval', type=int, default=100)
    parser.add_argument('--densify_grad_thresh', type=float, default=1e-3)
    parser.add_argument('--prune_opacity_thresh', type=float, default=0.01)
    parser.add_argument('--size_threshold', type=float, default=2.0)
    parser.add_argument('--opacity_reset_interval', type=int, default=1000)
    parser.add_argument('--densify_topk_ratio', type=float, default=0.05, help='Top-k ratio of grad to clone at each densify step')
    # SH growth
    parser.add_argument('--sh_increase_interval', type=int, default=1000)

    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    ensure_dir(args.out_dir)

    # Load sequence (images [F,3,H,W], cams list, times [F,1], masks [F,1,H,W] or None)
    images, cams, times, masks = load_sequence(args.data_root, mask_root=args.mask_root)
    F, C, H, W = images.shape

    # Initialize a tiny point cloud by sampling pixels from first frame with stratified sampling
    n_init = min(args.max_points, max(1024, (H * W) // 20))
    ys = torch.randint(0, H, (n_init,))
    xs = torch.randint(0, W, (n_init,))
    # Back-project onto a fronto-parallel plane at z=2.0 in cam 0, approximate world frame as cam0 frame
    fx, fy, cx, cy = cams[0]['K'][0,0], cams[0]['K'][1,1], cams[0]['K'][0,2], cams[0]['K'][1,2]
    z0 = 2.0
    X = (xs.float() - cx) * z0 / fx
    Y = (ys.float() - cy) * z0 / fy
    Z = torch.full_like(X, z0)
    xyz0 = torch.stack([X, Y, Z], dim=-1)
    rgb0 = images[0, :, ys, xs].permute(1,0).contiguous()  # [N,3]
    t0 = torch.zeros(n_init, 1)

    model = GaussianModel4D(sh_degree=args.sh_degree, device=device)
    model.init_from_pcd(xyz0.to(device), rgb0.to(device), t0.to(device))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Buffers for densification
    max_radii2D = torch.zeros(model.primitive._xyz.shape[0], device=device)
    grad_accum = torch.zeros(model.primitive._xyz.shape[0], device=device)
    vis_count = torch.zeros(model.primitive._xyz.shape[0], device=device)

    # SH growth
    active_sh_degree = min(0, args.sh_degree)
    
    # Create renderer
    if args.renderer == 'fast':
        renderer = create_renderer(
            renderer_type='fast',
            image_height=H,
            image_width=W,
            device=device,
            use_elliptical=False,
            use_quaternions=True
        )

    # Initialize temporal consistency loss
    temporal_loss_fn = TemporalConsistencyLoss(
        velocity_smooth_weight=1.0,
        position_smooth_weight=0.5,
        appearance_weight=0.2
    ) if args.w_temporal > 0 else None
    
    pbar = tqdm(range(1, args.iters + 1), desc='train')
    for it in pbar:
        optim.zero_grad()
        # Pick a random frame
        fidx = torch.randint(0, F, (1,)).item()
        gt = images[fidx].to(device)
        K = cams[fidx]['K'].to(device)
        R = cams[fidx]['R'].to(device)
        tt = cams[fidx]['t'].to(device)
        time = float(times[fidx, 0].item())
        mk = masks[fidx].to(device) if masks is not None else None

        # Viewing direction approximated as pointing towards camera center in world coordinates
        cam_pos = (-R.T @ tt).to(device)  # c2w translation
        dirs = (cam_pos[None, :] - model.xyz)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-6)  # [N,3]

        # Use time-dependent positions x(t) = x + v * t
        xyz_t = model.position_at_time(time)
        if args.renderer == 'fast':
            # Use fast renderer with CUDA acceleration (fallback to naive if unavailable)
            outputs = renderer(
                positions=xyz_t,
                scales=model.scales,
                rotations=model.quat,
                colors=model.get_rgb(dirs, active_sh_degree if args.sh_degree>0 else 0),
                opacities=model.opacity,
                camera_position=-R.T @ tt,
                camera_rotation=R.T,
                camera_intrinsics=K
            )
            img = outputs['rgb'].permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            a = outputs.get('visibility', torch.ones(H, W, device=device))
            d = torch.zeros(H, W, device=device)  # Depth not computed by fast renderer
        else:
            img, a, d = forward_splat(
                xyz_t, model.t, model.scales, model.scale_t, model.opacity.squeeze(-1), model.rgb_sh,
                dirs, active_sh_degree if args.sh_degree>0 else 0, K, R, tt, H, W, time, quat=model.quat, elliptical=True
            )

        # Losses
        L_l1 = l1_loss(img, gt)
        L_ssim = 1.0 - ssim(img, gt)
        loss = (1.0 - args.w_ssim) * L_l1 + args.w_ssim * L_ssim

        if mk is not None and args.w_opa_mask > 0:
            sky = (1.0 - (mk>0.5).float())  # 1 for background
            o = a.clamp(1e-6, 1-1e-6)
            L_opa = (- sky * torch.log(1 - o)).mean()
            loss = loss + args.w_opa_mask * L_opa
        
        # Add temporal consistency loss
        if args.w_temporal > 0 and F > 1:
            # Sample a temporal window
            window_size = min(args.temporal_window, F)
            if fidx + window_size <= F:
                time_indices = torch.arange(fidx, min(fidx + window_size, F))
            else:
                time_indices = torch.arange(max(0, fidx - window_size + 1), fidx + 1)
            
            if len(time_indices) > 1:
                # Get timestamps for the window
                window_times = times[time_indices, 0].to(device)
                
                # Compute temporal regularization
                L_temporal = temporal_consistency_regularizer(
                    model,
                    window_times,
                    base_weight=1.0,
                    use_knn=True,
                    k_neighbors=5
                )
                loss = loss + args.w_temporal * L_temporal

        # Backprop
        loss.backward()

        # Densify and prune
        with torch.no_grad():
            # Update max_radii2D approx from current r_pix computation
            # recompute r_pix
            # Rough 2D footprint size from scale and depth (same as renderer)
            uv, z, _ = None, None, None
            # quick recompute inline (duplicated logic minimal):
            Xc = (R @ model.xyz.T).T + tt[None]
            z = Xc[:, 2].clamp(min=1e-6)
            fx = K[0,0]; fy = K[1,1]
            r_pix = 0.5 * (fx * model.scales[:,0] / z + fy * model.scales[:,1] / z)
            max_radii2D = torch.maximum(max_radii2D, r_pix)

            if args.densify_from_iter <= it <= args.densify_until_iter and it % args.densification_interval == 0:
                # Approximate visibility and accumulate gradient
                x = Xc[:,0] / z; y = Xc[:,1] / z
                uv = torch.stack([(K[0,0]*x + K[0,2]), (K[1,1]*y + K[1,2])], dim=-1)
                vis = (uv[:,0] >= 0) & (uv[:,0] < W) & (uv[:,1] >= 0) & (uv[:,1] < H) & (z > 0)
                vis_count[:model.xyz.shape[0]] += vis.float()
                if model.xyz.grad is not None:
                    gnorm = model.xyz.grad.norm(dim=1)
                    grad_accum[:model.xyz.shape[0]] += gnorm

                # Clone top-k ratio by accumulated grad
                N = model.xyz.shape[0]
                if N > 0:
                    k = max(1, int(args.densify_topk_ratio * N))
                    vals, idxs = torch.topk(grad_accum[:N], k)
                    clone_mask = torch.zeros(N, dtype=torch.bool, device=device)
                    clone_mask[idxs] = True
                    if clone_mask.any():
                        model.densify_clone(clone_mask)
                        added = int(clone_mask.sum().item())
                        max_radii2D = torch.cat([max_radii2D, torch.zeros(added, device=device)], dim=0)
                        grad_accum = torch.cat([grad_accum, torch.zeros(added, device=device)], dim=0)
                        vis_count = torch.cat([vis_count, torch.zeros(added, device=device)], dim=0)
                    grad_accum[:N] *= 0.0
                    vis_count[:N] *= 0.0

                # Prune low-opacity AND low-vis AND small points
                opa = model.opacity.squeeze(-1).detach()
                small = max_radii2D < args.size_threshold
                lowvis = vis_count < (args.densification_interval * 0.25)
                prune_mask = (opa < args.prune_opacity_thresh) & small & lowvis
                if prune_mask.any():
                    keep = ~prune_mask
                    model.prune(keep)
                    max_radii2D = max_radii2D[keep]
                    grad_accum = grad_accum[keep]
                    vis_count = vis_count[keep]

            if it % args.opacity_reset_interval == 0 and it < args.densify_until_iter:
                # Reset opacity towards mid-range to allow further growth
                model.primitive._opacity.data.clamp_(0.1, 0.9)

        optim.step()

        with torch.no_grad():
            metric = psnr(img, gt).item()
        pbar.set_postfix({'l1': L_l1.item(), 'psnr': f'{metric:.2f}'})

        # SH growth
        if args.sh_degree > 0 and it % args.sh_increase_interval == 0:
            active_sh_degree = min(args.sh_degree, active_sh_degree + 1)

        if it % args.save_every == 0:
            torch.save({'state': model.state_dict_compact()}, os.path.join(args.out_dir, f'model_{it}.pt'))

        if it % args.validate_every == 0:
            # quick validation on first frame
            with torch.no_grad():
                K0 = cams[0]['K'].to(device)
                R0 = cams[0]['R'].to(device)
                t0c = cams[0]['t'].to(device)
                vdirs = ((-R0.T @ t0c)[None, :] - model.xyz)
                vdirs = vdirs / (vdirs.norm(dim=-1, keepdim=True) + 1e-6)
                tval = float(times[0,0].item())
                xyz_t0 = model.position_at_time(tval)
                vimg, va, vd = forward_splat(
                    xyz_t0, model.t, model.scales, model.scale_t, model.opacity.squeeze(-1), model.rgb_sh,
                    vdirs, active_sh_degree if args.sh_degree>0 else 0, K0, R0, t0c, H, W, tval, quat=model.quat, elliptical=True
                )
                image_np = (vimg.clamp(0,1).cpu().numpy().transpose(1,2,0) * 255).astype('uint8')
                import imageio.v2 as imageio
                imageio.imwrite(os.path.join(args.out_dir, f'val_{it:05d}.png'), image_np)

    # Final export
    torch.save({'state': model.state_dict_compact()}, os.path.join(args.out_dir, f'model_final.pt'))


if __name__ == '__main__':
    main()
