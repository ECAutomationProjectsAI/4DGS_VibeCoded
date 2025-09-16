import os
import argparse
import torch
import imageio.v2 as imageio
from gs4d.losses import psnr
from gs4d.dataio import load_sequence


def main():
    parser = argparse.ArgumentParser(description='Evaluate renders against ground truth')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--renders_dir', type=str, required=True)
    args = parser.parse_args()

    images, cams, times = load_sequence(args.data_root)
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
