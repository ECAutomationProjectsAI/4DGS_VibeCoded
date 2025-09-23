#!/usr/bin/env python3
"""
DEPRECATED: Unified preprocessor has been replaced by the three-step pipeline.

Use these instead (RunPod/Ubuntu only):
  1) python3 scripts/01_extract_and_map.py /workspace/videos -o /workspace/dataset --resize 1280 720
  2) python3 scripts/02_calibrate_cameras.py --data_root /workspace/dataset
  3) python3 scripts/03_train_4dgs.py --data_root /workspace/dataset --out_dir /workspace/outputs/exp
"""

if __name__ == '__main__':
    print(__doc__)
    import sys
    sys.exit(1)
