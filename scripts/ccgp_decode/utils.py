# ccgp_decode/utils.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional

def remap_labels_to_contiguous(y, classes=None):
    """
    Remap labels to contiguous 0..K-1.
    If `classes` provided, uses that ordering.
    """
    y = np.asarray(y).astype(int)

    if classes is None:
        classes = np.unique(y)
    else:
        classes = np.asarray(classes).astype(int)
        if not np.all(np.isin(y, classes)):
            missing = np.setdiff1d(np.unique(y), classes)
            raise RuntimeError(f"Labels contain values not in provided classes: {missing.tolist()}")

    mapping = {int(c): int(i) for i, c in enumerate(classes)}
    y_new = np.array([mapping[int(v)] for v in y], dtype=int)
    return y_new, mapping, classes