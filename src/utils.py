"""Small shared utilities used across pipeline scripts."""

import json
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories(*paths: Path) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    """Write a dictionary to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

