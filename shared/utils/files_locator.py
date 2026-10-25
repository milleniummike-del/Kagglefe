"""
Checkpoints and file path locator helper for Wan2GP and LTX models.
"""

import os
from typing import List, Optional

_checkpoints_paths = ["models", "ckpts", "/kaggle/tmp/models", "."]


def set_checkpoints_paths(paths: List[str]):
    global _checkpoints_paths
    _checkpoints_paths = list(paths)


def get_checkpoints_paths() -> List[str]:
    return _checkpoints_paths


def find_file(filename: str) -> Optional[str]:
    for p in _checkpoints_paths:
        candidate = os.path.join(p, filename)
        if os.path.isfile(candidate):
            return candidate
    return None
