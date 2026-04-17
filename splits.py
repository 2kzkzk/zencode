"""Data split utilities (few-shot node classification).

This file exists so that causal_main.py does not depend on any baseline code.
"""

from __future__ import annotations

import random
from typing import List, Tuple


def create_splits(Y, k: int) -> Tuple[List[int], List[int], List[int]]:
    """Generate train/validation/test indices for few-shot node classification.

    Each class contributes exactly k labeled nodes for training.

    Args:
        Y: 1D tensor / list-like of integer labels of length N.
        k: number of labeled samples per class in the training set.

    Returns:
        (train_idx, val_idx, test_idx): sorted index lists.
    """
    label_to_indices = {}
    for idx, label in enumerate(Y):
        label = int(label)
        label_to_indices.setdefault(label, []).append(idx)

    train_indices = set()
    val_indices = set()

    for label, indices in label_to_indices.items():
        n = len(indices)
        if n <= k:
            train = indices
            val = []
        else:
            sampled = random.sample(indices, min(n, 2 * k))
            train = sampled[:k]
            val = sampled[k:]

        train_indices.update(train)
        val_indices.update(val)

    all_indices = set(range(len(Y)))
    test_indices = all_indices - train_indices - val_indices

    return sorted(train_indices), sorted(val_indices), sorted(test_indices)
