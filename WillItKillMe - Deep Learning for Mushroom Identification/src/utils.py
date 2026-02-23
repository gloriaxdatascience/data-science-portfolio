# src/utils.py
import os
import random
import numpy as np
import torch
from pathlib import Path


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
def get_data_paths(base_dir=".."):
    base = Path(base_dir).resolve()
    data = base / "data" / "raw"
    return {
        "train": data / "train",
        "validate": data / "validate",
        "test": data / "test",
    }
