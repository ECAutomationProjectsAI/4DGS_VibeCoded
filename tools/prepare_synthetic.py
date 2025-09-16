import os
import argparse
import numpy as np
import imageio.v2 as imageio

"""
Generate a minimal synthetic dynamic sequence for validation:
- Rigid moving colored square on a black background
- Camera: pinhole, static at origin looking towards +Z
- Outputs: frames/*.png and transforms.json
"""

def main():
    parser = argparse.ArgumentParser(description='Create a tiny synthetic dynamic dataset')
    parser.add_argument('--out_root', type=str, default='data/synth_tiny')
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--H', type=int, default=128)
    parser.add_argument('--W', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_root, 'frames'), exist_ok=True)

    H, W = args.H, args.W
    fx = fy = 100.0
    cx, cy = W/2.0, H/2.0

    frames_meta = []
    for i in range(args.frames):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        # moving square
        size = H // 8
        u0 = int(W*0.25 + i * (W*0.5 / max(1,args.frames-1)))
        v0 = int(H*0.25 + i * (H*0.3 / max(1,args.frames-1)))
        u1 = min(W-1, u0 + size)
        v1 = min(H-1, v0 + size)
        color = (int(255*i/max(1,args.frames-1)), 64, int(255 - 255*i/max(1,args.frames-1)))
        img[v0:v1, u0:u1] = np.array(color, dtype=np.uint8)
        fp = f'frames/{i:05d}.png'
        imageio.imwrite(os.path.join(args.out_root, fp), img)
        # Static c2w at identity, but we store 4x4 c2w (camera at origin looking +Z, OpenGL-style)
        c2w = np.eye(4, dtype=np.float32)
        frames_meta.append({
            'file_path': fp,
            'transform_matrix': c2w.tolist(),
            'time': float(i)
        })

    meta = {
        'fl_x': fx,
        'fl_y': fy,
        'cx': cx,
        'cy': cy,
        'h': H,
        'w': W,
        'frames': frames_meta
    }

    import json
    with open(os.path.join(args.out_root, 'transforms.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Wrote synthetic dataset to {args.out_root}')


if __name__ == '__main__':
    main()
