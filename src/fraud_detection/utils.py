import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    """Save a dictionary as a pretty JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

