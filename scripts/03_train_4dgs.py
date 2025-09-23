#!/usr/bin/env python3
"""
Step 3: Train 4D Gaussian Splatting using the prepared dataset (transforms.json).

This is a thin wrapper around tools/train.py with safer defaults to avoid CUDA OOM.

Usage (RunPod/Ubuntu):
  python3 scripts/03_train_4dgs.py --data_root /workspace/dataset --out_dir /workspace/outputs/exp \
      --iters 30000 --renderer fast --w_temporal 0.01

Notes:
- Run after:
  1) scripts/01_extract_and_map.py
  2) scripts/02_calibrate_cameras.py
- Exposes a subset of tools/train.py options and passes the rest through.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Project root to build a robust call to tools/train.py
ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = ROOT / 'tools' / 'train.py'


def main():
    ap = argparse.ArgumentParser(description='Step 3: Train 4D Gaussian Splatting (wrapper)')
    ap.add_argument('--data_root', type=str, required=True, help='Dataset root containing transforms.json (from Step 2)')
    ap.add_argument('--out_dir', type=str, required=True, help='Output dir for checkpoints and renders')

    # Safer defaults to reduce OOM. Every option can be overridden via --extra.
    ap.add_argument('--iters', type=int, default=30000)
    ap.add_argument('--renderer', type=str, default='fast', choices=['fast','naive'])
    ap.add_argument('--sh_degree', type=int, default=3)

    # Memory & warmup
    ap.add_argument('--vram_fraction', type=float, default=0.5, help='Fraction of GPU VRAM to use (default 0.5)')
    ap.add_argument('--max_points', type=int, default=-1, help='Max Gaussians (-1 = auto based on VRAM*fraction)')
    ap.add_argument('--max_memory_gb', type=float, default=-1, help='Max CPU RAM for images (-1 = auto)')
    ap.add_argument('--warmup_iters', type=int, default=500)
    ap.add_argument('--warmup_points', type=int, default=200000)
    ap.add_argument('--warmup_downscale', type=int, default=2)

    # Temporal loss
    ap.add_argument('--w_temporal', type=float, default=0.01)
    ap.add_argument('--temporal_window', type=int, default=3)

    # Pass-through extra args
    ap.add_argument('--extra', nargs=argparse.REMAINDER, help='Additional args passed to tools/train.py verbatim')

    args = ap.parse_args()

    root = Path(args.data_root)
    if not (root / 'transforms.json').exists():
        print(f"ERROR: transforms.json not found under {root}. Run scripts/02_calibrate_cameras.py first.")
        sys.exit(1)

    # Build command
    cmd = [
        sys.executable, str(TRAIN_PY),
        '--data_root', str(root),
        '--out_dir', str(args.out_dir),
        '--iters', str(args.iters),
        '--renderer', args.renderer,
        '--sh_degree', str(args.sh_degree),
        '--vram_fraction', str(args.vram_fraction),
        '--max_points', str(args.max_points),
        '--max_memory_gb', str(args.max_memory_gb),
        '--warmup_iters', str(args.warmup_iters),
        '--warmup_points', str(args.warmup_points),
        '--warmup_downscale', str(args.warmup_downscale),
        '--w_temporal', str(args.w_temporal),
        '--temporal_window', str(args.temporal_window),
    ]

    if args.extra:
        cmd.extend(args.extra)

    print("\nRunning:")
    print(' '.join(map(str, cmd)))

    # Launch training
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


if __name__ == '__main__':
    main()
