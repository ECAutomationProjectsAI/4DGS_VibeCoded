import os
import sys
import argparse
import torch
import imageio.v2 as imageio

# Add parent directory to path to import gs4d module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs4d.losses import psnr
from gs4d.dataio import load_sequence


def main():
    parser = argparse.ArgumentParser(description='Evaluate renders against ground truth')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--renders_dir', type=str, required=True)
    parser.add_argument('--start_frame', type=int, default=None, help='Start frame index (inclusive)')
    parser.add_argument('--end_frame', type=int, default=None, help='End frame index (exclusive)')
    args = parser.parse_args()

    images, cams, times = load_sequence(args.data_root, start_frame=args.start_frame, end_frame=args.end_frame)
    F, C, H, W = images.shape

    psnrs = []
    for f in range(F):
        gt = images[f]
        rp = os.path.join(args.renders_dir, f'{f:05d}.png')
        im = imageio.imread(rp)
        if im.shape[:2] != (H, W):
            raise ValueError(f'Render {rp} has shape {im.shape}, expected {(H,W)}')
        imt = torch.from_numpy(im).float().permute(2,0,1) / 255.0
        ps = psnr(imt, gt).item()
        psnrs.append(ps)
    print(f'Average PSNR over {F} frames: {sum(psnrs)/len(psnrs):.2f} dB')


if __name__ == '__main__':
    main()
