import os
import sys
import argparse
import torch
import imageio.v2 as imageio

# Add parent directory to path to import gs4d module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs4d.gaussians import GaussianModel4D
from gs4d.dataio import load_sequence
from gs4d.renderer import forward_splat
from gs4d.fast_renderer import forward_splat_fast


def main():
    parser = argparse.ArgumentParser(description='Render frames from a trained 4DGS model')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='renders')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--renderer', type=str, default='naive', choices=['naive','fast'])
    parser.add_argument('--start_frame', type=int, default=None, help='Start frame index (inclusive)')
    parser.add_argument('--end_frame', type=int, default=None, help='End frame index (exclusive)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt['state']

    model = GaussianModel4D(sh_degree=args.sh_degree, device=device)
    model.load_state_dict_compact(state)

    images, cams, times = load_sequence(args.data_root, start_frame=args.start_frame, end_frame=args.end_frame)
    F, C, H, W = images.shape

    for f in range(F):
        K = cams[f]['K'].to(device)
        R = cams[f]['R'].to(device)
        t = cams[f]['t'].to(device)
        time = float(times[f,0].item())
        cam_pos = (-R.T @ t).to(device)
        dirs = (cam_pos[None, :] - model.xyz).detach()
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-6)
        xyz_t = model.position_at_time(time)
        with torch.no_grad():
            if args.renderer == 'fast':
                img, a, d = forward_splat_fast(xyz_t, model.t, model.scales, model.scale_t, model.opacity.squeeze(-1), model.rgb_sh,
                                               dirs, args.sh_degree, K, R, t, H, W, time, quat=model.quat)
            else:
                img, a, d = forward_splat(xyz_t, model.t, model.scales, model.scale_t, model.opacity.squeeze(-1), model.rgb_sh,
                                          dirs, args.sh_degree, K, R, t, H, W, time, quat=model.quat, elliptical=True)
        im = (img.clamp(0,1).cpu().numpy().transpose(1,2,0) * 255).astype('uint8')
        imageio.imwrite(os.path.join(args.out_dir, f'{f:05d}.png'), im)


if __name__ == '__main__':
    main()
