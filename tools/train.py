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


def _reduce_lr(optim, factor=0.5, min_lr=1e-5):
    for g in optim.param_groups:
        new_lr = max(min_lr, g['lr'] * factor)
        g['lr'] = new_lr
    return optim.param_groups[0]['lr']


def _prune_nonfinite_gaussians(model) -> int:
    """Remove Gaussians with non-finite parameters or invalid scales."""
    p = model.primitive
    tensors = [p._xyz, p._vel, p._t, p._log_scale, p._log_scale_t, p._quat, p._omega_t, p._opacity, p._rgb_sh]
    # Build keep mask across all tensors
    keep_masks = []
    for t in tensors:
        if t is None:
            continue
        km = torch.isfinite(t)
        # Reduce over parameter dims
        km = km.view(km.shape[0], -1).all(dim=1)
        keep_masks.append(km)
    if not keep_masks:
        return 0
    keep = torch.stack(keep_masks, dim=0).all(dim=0)
    # Also enforce positive scales after exp: log_scale/log_scale_t finite is enough; clamp raw logs to reasonable range
    p._log_scale.data.clamp_(-6.0, 2.0)
    p._log_scale_t.data.clamp_(-8.0, 2.0)
    # Re-normalize quaternions to avoid drift
    p._quat.data = p._quat.data / (p._quat.data.norm(dim=-1, keepdim=True) + 1e-8)
    # Prune if needed
    removed = int((~keep).sum().item())
    if removed > 0:
        model.prune(keep)
    return removed


def _nan_inf_found(*tensors) -> bool:
    for t in tensors:
        if t is None:
            continue
        if not torch.isfinite(t).all():
            return True
    return False


def get_available_memory():
    """Get available system and GPU memory."""
    import psutil
    
    # Get system RAM
    mem = psutil.virtual_memory()
    available_ram_gb = mem.available / (1024**3)
    total_ram_gb = mem.total / (1024**3)
    
    # Get GPU memory if available
    available_vram_gb = 0
    total_vram_gb = 0
    if torch.cuda.is_available():
        # Get free GPU memory
        free_vram, total_vram = torch.cuda.mem_get_info()
        available_vram_gb = free_vram / (1024**3)
        total_vram_gb = total_vram / (1024**3)
    
    return available_ram_gb, total_ram_gb, available_vram_gb, total_vram_gb

def main():
    parser = argparse.ArgumentParser(description='Train 4D Gaussian Splatting (from scratch)')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--mask_root', type=str, default=None, help='Optional mask folder; mask filenames should match image basenames')
    parser.add_argument('--out_dir', type=str, default='outputs/exp')
    parser.add_argument('--iters', type=int, default=30000, help='Training iterations (more = better quality)')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=int, default=-1, help='GPU ID to use (-1 for auto-select best GPU)')
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--validate_every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_points', type=int, default=-1, help='Maximum Gaussians (-1 for auto based on GPU memory)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--max_frames', type=int, default=-1, help='Maximum frames to load (-1 for auto)')
    parser.add_argument('--start_frame', type=int, default=None, help='Start frame index (inclusive)')
    parser.add_argument('--end_frame', type=int, default=None, help='End frame index (exclusive)')
    parser.add_argument('--max_memory_gb', type=float, default=-1, help='Maximum RAM for images (-1 for auto 90% of available)')
    parser.add_argument('--memory_fraction', type=float, default=0.90, help='Fraction of available memory to use (0.90 = 90%)')
    parser.add_argument('--vram_fraction', type=float, default=0.95, help='Fraction of GPU memory to use (0.95 = 95%)')
    parser.add_argument('--renderer', type=str, default='fast', choices=['naive','fast'], help='Renderer backend (fast recommended)')
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
    
    # Import psutil for memory monitoring
    import psutil
    
    # Get available resources
    available_ram_gb, total_ram_gb, available_vram_gb, total_vram_gb = get_available_memory()
    
    print("\n" + "="*60)
    print("           4D Gaussian Splatting Training")
    print("="*60)
    
    # Auto-detect and configure GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        
        if args.gpu_id == -1:
            # Auto-select GPU with most free memory
            best_gpu = 0
            best_free_mem = 0
            
            for i in range(num_gpus):
                torch.cuda.set_device(i)
                free_mem, _ = torch.cuda.mem_get_info(i)
                if free_mem > best_free_mem:
                    best_free_mem = free_mem
                    best_gpu = i
            
            args.gpu_id = best_gpu
            torch.cuda.set_device(args.gpu_id)
            print(f"\n🚀 Auto-selected GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
        else:
            # Use specified GPU
            torch.cuda.set_device(args.gpu_id)
            print(f"\n🎯 Using specified GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
        
        # Update available VRAM for the selected GPU
        free_vram, total_vram = torch.cuda.mem_get_info(args.gpu_id)
        available_vram_gb = free_vram / (1024**3)
        total_vram_gb = total_vram / (1024**3)
    else:
        print("\n⚠️  No GPU available, using CPU (will be slow)")
    
    # Display system resources
    print(f"\n📊 System Resources Detected:")
    print(f"   CPU RAM: {available_ram_gb:.1f}/{total_ram_gb:.1f} GB available ({available_ram_gb/total_ram_gb*100:.1f}% free)")
    if torch.cuda.is_available():
        print(f"   GPU VRAM: {available_vram_gb:.1f}/{total_vram_gb:.1f} GB available ({available_vram_gb/total_vram_gb*100:.1f}% free)")
    
    # Resource validation - check minimums
    MIN_RAM_GB = 4.0  # Minimum 4GB RAM needed
    MIN_VRAM_GB = 2.0  # Minimum 2GB VRAM for GPU training
    
    if available_ram_gb < MIN_RAM_GB:
        print(f"\n❌ ERROR: Insufficient RAM!")
        print(f"   Available: {available_ram_gb:.1f} GB")
        print(f"   Required: {MIN_RAM_GB:.1f} GB minimum")
        print(f"\n💡 Solutions:")
        print(f"   1. Close other applications to free memory")
        print(f"   2. Reduce dataset size with --max_frames")
        print(f"   3. Use lower resolution images")
        sys.exit(1)
    
    if torch.cuda.is_available() and available_vram_gb < MIN_VRAM_GB:
        print(f"\n❌ ERROR: Insufficient GPU VRAM!")
        print(f"   Available: {available_vram_gb:.1f} GB")
        print(f"   Required: {MIN_VRAM_GB:.1f} GB minimum")
        print(f"\n💡 Solutions:")
        print(f"   1. Close other GPU applications")
        print(f"   2. Use --max_points to limit Gaussians")
        print(f"   3. Use CPU training with --device cpu (slow)")
        sys.exit(1)
    
    # Auto-configure memory limits based on available resources
    if args.max_memory_gb == -1:
        # Use 90% of available RAM by default
        args.max_memory_gb = available_ram_gb * args.memory_fraction
        print(f"\n🔧 Auto-configured RAM usage: {args.max_memory_gb:.1f} GB ({args.memory_fraction*100:.0f}% of {available_ram_gb:.1f} GB)")
    else:
        # Validate user-specified memory
        if args.max_memory_gb > available_ram_gb:
            print(f"\n⚠️  WARNING: Requested {args.max_memory_gb:.1f} GB exceeds available {available_ram_gb:.1f} GB")
            print(f"   Capping at {available_ram_gb * 0.95:.1f} GB (95% of available)")
            args.max_memory_gb = available_ram_gb * 0.95
        print(f"\n📌 Using specified memory limit: {args.max_memory_gb:.1f} GB")
    
    # Auto-configure max points based on GPU memory
    if args.max_points == -1 and torch.cuda.is_available():
        # More aggressive: 150k points per GB with 95% VRAM usage
        vram_for_points = available_vram_gb * args.vram_fraction
        args.max_points = min(int(vram_for_points * 150_000), 2_000_000)  # Cap at 2M points
        print(f"🔧 Auto-configured max points: {args.max_points:,} (using {vram_for_points:.1f} GB VRAM at {args.vram_fraction*100:.0f}%)")
    elif args.max_points == -1:
        # CPU fallback
        args.max_points = 25_000
        print(f"🔧 Using default max points for CPU: {args.max_points:,}")
    else:
        print(f"📌 Using specified max points: {args.max_points:,}")
    
    # Auto-configure training iterations based on complexity
    if args.iters == 30000 and args.max_points > 500_000:
        # For complex scenes, increase iterations
        args.iters = 50000
        print(f"🔧 Auto-increased iterations to {args.iters:,} for complex scene")
    
    print(f"\n🎲 Random seed: {args.seed}")
    print(f"🎨 Renderer: {args.renderer}")
    print(f"🔄 Training iterations: {args.iters:,}")
    print(f"📂 Data root: {args.data_root}")
    print(f"💾 Output directory: {args.out_dir}")
    print("\n" + "="*60)
    
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    ensure_dir(args.out_dir)
    
    print("\nLoading dataset...")
    # Check if data exists
    if not os.path.exists(args.data_root):
        print(f"ERROR: Data root does not exist: {args.data_root}")
        sys.exit(1)
        
    transforms_path = os.path.join(args.data_root, 'transforms.json')
    if not os.path.exists(transforms_path):
        print(f"ERROR: transforms.json not found in {args.data_root}")
        print("Please run preprocessing first: python tools/preprocess_video.py")
        sys.exit(1)

    # Load sequence with memory limits
    try:
        images, cams, times, masks = load_sequence(
            args.data_root, 
            mask_root=args.mask_root,
            max_frames=args.max_frames,
            max_memory_gb=args.max_memory_gb,
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )
        F, C, H, W = images.shape
        print(f"\nSuccessfully loaded {F} frames, resolution: {W}x{H}")
        print(f"Images tensor size: {images.shape}, dtype: {images.dtype}")
        print(f"Memory usage: {images.element_size() * images.numel() / 1e9:.2f} GB")
        
        # Move to GPU in chunks if needed
        if device.type == 'cuda':
            # Don't move all images to GPU at once
            print(f"\nGPU Memory before: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            images = images.to(device)
            print(f"GPU Memory after moving images: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            
    except Exception as e:
        print(f"\nERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Initialize a tiny point cloud by sampling pixels from first frame
    print("\nInitializing Gaussians...")
    n_init = min(args.max_points // 10, max(1024, (H * W) // 100))  # Start with fewer points
    print(f"  Initial points: {n_init} (will grow to max {args.max_points})")
    
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
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Training for {args.iters:,} iterations")
    print(f"Saving checkpoints every {args.save_every:,} iterations")
    print(f"Validation every {args.validate_every:,} iterations")
    print(f"Densification every {args.densification_interval} iterations (until iter {args.densify_until_iter:,})")
    if device.type == 'cuda':
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(device) / 1e9:.2f}/{total_vram_gb:.1f} GB")
    print("\nStarting optimization...\n")
    
    # Track metrics
    best_psnr = 0
    loss_history = []
    memory_warnings = 0
    
    def _clamp_model_sanity():
        p = model.primitive
        # Keep logs in reasonable bounds to avoid degenerate scales
        p._log_scale.data.clamp_(-6.0, 2.0)
        p._log_scale_t.data.clamp_(-8.0, 2.0)
        # Re-normalize quaternions
        p._quat.data = p._quat.data / (p._quat.data.norm(dim=-1, keepdim=True) + 1e-8)
        # Opacity raw parameters: keep in finite range
        p._opacity.data = p._opacity.data.clamp(-6.0, 6.0)
        # Replace any NaNs in SH with zeros
        nan_mask = ~torch.isfinite(p._rgb_sh.data)
        if nan_mask.any():
            p._rgb_sh.data[nan_mask] = 0.0

    def _render_with_guard(xyz_t, dirs, K, R, tt, H, W, time, active_sh_degree):
        # Try fast renderer first
        if args.renderer == 'fast':
            try:
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
                img_fast = outputs['rgb'].permute(2, 0, 1)
                if torch.isfinite(img_fast).all():
                    a_fast = outputs.get('visibility', torch.ones(H, W, device=device))
                    d_fast = torch.zeros(H, W, device=device)
                    return img_fast, a_fast, d_fast, 'fast'
                else:
                    print("\n⚠️ Fast renderer produced non-finite values; falling back to naive for this iter")
            except Exception as e:
                print(f"\n⚠️ Fast renderer exception: {e}. Falling back to naive for this iter")
        # Naive fallback (elliptical on)
        img_n, a_n, d_n = forward_splat(
            xyz_t, model.t, model.scales, model.scale_t, model.opacity.squeeze(-1), model.rgb_sh,
            dirs, active_sh_degree if args.sh_degree>0 else 0, K, R, tt, H, W, time, quat=model.quat, elliptical=True
        )
        return img_n, a_n, d_n, 'naive'

    pbar = tqdm(range(1, args.iters + 1), desc='Training', unit='iter')
    for it in pbar:
        optim.zero_grad()
        _clamp_model_sanity()
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
        img, a, d, used = _render_with_guard(xyz_t, dirs, K, R, tt, H, W, time, active_sh_degree)
        if not torch.isfinite(img).all():
            print(f"\n❌ Non-finite image tensor right after rendering (mode={used}). Recovering…")
            pruned = _prune_nonfinite_gaussians(model)
            new_lr = _reduce_lr(optim, factor=0.5)
            print(f"   - Pruned invalid Gaussians: {pruned}")
            print(f"   - Reduced learning rate to: {new_lr:.2e}")
            optim.zero_grad(set_to_none=True)
            continue

        # Losses
        L_l1 = l1_loss(img, gt)
        L_ssim = 1.0 - ssim(img, gt)
        loss = (1.0 - args.w_ssim) * L_l1 + args.w_ssim * L_ssim

        # NaN/Inf watchdog before backward
        if _nan_inf_found(loss, img, model.xyz, model.scales, model.opacity):
            print(f"\n❌ Detected non-finite values at iter {it}. Engaging recovery…")
            pruned = _prune_nonfinite_gaussians(model)
            new_lr = _reduce_lr(optim, factor=0.5)
            print(f"   - Pruned invalid Gaussians: {pruned}")
            print(f"   - Reduced learning rate to: {new_lr:.2e}")
            # Skip this iteration safely
            optim.zero_grad(set_to_none=True)
            continue

        if mk is not None and args.w_opa_mask > 0:
            sky = (1.0 - (mk>0.5).float())  # 1 for background
            o = a.clamp(1e-6, 1-1e-6)
            L_opa = (- sky * torch.log(1 - o)).mean()
            loss = loss + args.w_opa_mask * L_opa
        
        # Add temporal consistency loss
        if args.w_temporal > 0 and F > 1:
            # Disable temporal loss if too many points (to avoid OOM)
            if model.xyz.shape[0] > 50000:
                # Skip temporal loss for large point clouds
                L_temporal = 0.0
            else:
                # Sample a temporal window
                window_size = min(args.temporal_window, F)
                if fidx + window_size <= F:
                    time_indices = torch.arange(fidx, min(fidx + window_size, F))
                else:
                    time_indices = torch.arange(max(0, fidx - window_size + 1), fidx + 1)
                
                if len(time_indices) > 1:
                    # Get timestamps for the window
                    window_times = times[time_indices, 0].to(device)
                    
                    # Compute temporal regularization with reduced neighbors
                    try:
                        L_temporal = temporal_consistency_regularizer(
                            model,
                            window_times,
                            base_weight=1.0,
                            use_knn=True,
                            k_neighbors=3  # Reduced from 5 to save memory
                        )
                        loss = loss + args.w_temporal * L_temporal
                    except torch.cuda.OutOfMemoryError:
                        # If still OOM, skip temporal loss
                        print(f"\n⚠️ WARNING: Skipping temporal loss due to memory constraints")
                        L_temporal = 0.0

        # Backprop (guard)
        if not torch.isfinite(loss):
            print(f"\n❌ Non-finite loss at iter {it}. Skipping backward and recovering…")
            pruned = _prune_nonfinite_gaussians(model)
            new_lr = _reduce_lr(optim, factor=0.5)
            print(f"   - Pruned invalid Gaussians: {pruned}")
            print(f"   - Reduced learning rate to: {new_lr:.2e}")
            optim.zero_grad(set_to_none=True)
            continue
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

        # Update progress bar with comprehensive metrics
        with torch.no_grad():
            current_psnr = psnr(img, gt).item()
            best_psnr = max(best_psnr, current_psnr)
            loss_history.append(loss.item())
            
            # Memory monitoring every 100 iterations
            if it % 100 == 0 and device.type == 'cuda':
                current_vram = torch.cuda.memory_allocated(device) / 1e9
                vram_percent = (current_vram / total_vram_gb) * 100
                
                # Check for memory warnings
                if vram_percent > 95:
                    memory_warnings += 1
                    if memory_warnings > 5:
                        print(f"\n⚠️ WARNING: High VRAM usage ({vram_percent:.1f}%)!")
                        print(f"   Consider reducing --max_points or --max_memory_gb")
                        memory_warnings = 0  # Reset counter
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PSNR': f'{current_psnr:.2f}',
                    'Best': f'{best_psnr:.2f}',
                    'Points': f'{model.xyz.shape[0]:,}',
                    'VRAM': f'{current_vram:.1f}GB ({vram_percent:.0f}%)'
                })
            else:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PSNR': f'{current_psnr:.2f}',
                    'Best': f'{best_psnr:.2f}',
                    'Points': f'{model.xyz.shape[0]:,}'
                })

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
    
    # Training summary
    print("\n" + "="*60)
    print("✅ Training Completed Successfully!")
    print("="*60)
    print(f"\n📊 Final Statistics:")
    print(f"   Total iterations: {args.iters:,}")
    print(f"   Final point count: {model.xyz.shape[0]:,}")
    print(f"   Best PSNR achieved: {best_psnr:.2f} dB")
    if len(loss_history) > 0:
        print(f"   Final loss: {loss_history[-1]:.4f}")
        print(f"   Average loss: {np.mean(loss_history):.4f}")
    
    if device.type == 'cuda':
        final_vram = torch.cuda.memory_allocated(device) / 1e9
        print(f"\n💾 Memory Usage:")
        print(f"   Peak VRAM: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
        print(f"   Final VRAM: {final_vram:.2f} GB")
    
    print(f"\n📂 Output Files:")
    print(f"   Model saved to: {os.path.join(args.out_dir, 'model_final.pt')}")
    checkpoint_files = [f for f in os.listdir(args.out_dir) if f.startswith('model_') and f.endswith('.pt')]
    print(f"   Checkpoints saved: {len(checkpoint_files)}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Render results: python tools/render.py --data_root {args.data_root} --ckpt {os.path.join(args.out_dir, 'model_final.pt')} --out_dir renders/")
    print(f"   2. Evaluate quality: python tools/evaluate.py --data_root {args.data_root} --renders_dir renders/")
    print(f"   3. Create video: python tools/create_video.py --input_dir renders/ --output video.mp4")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
