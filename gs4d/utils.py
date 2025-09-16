import os
import json
from dataclasses import dataclass
from typing import Optional


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


@dataclass
class TrainConfig:
    data_root: str
    out_dir: str
    iters: int = 5000
    lr: float = 1e-2
    sh_degree: int = 3
    H: Optional[int] = None
    W: Optional[int] = None
    save_every: int = 1000
    validate_every: int = 500
    seed: int = 42
    device: str = 'cuda'

