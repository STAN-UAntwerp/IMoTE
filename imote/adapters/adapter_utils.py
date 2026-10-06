import numpy as np

def split_by_threshold(X_train, current_indices, pivot_idx, pivot_value):
    mask = X_train[current_indices, pivot_idx] <= pivot_value
    return _apply_mask(current_indices, mask)

def split_by_categories(X_train, current_indices, pivot_idx, left_categories):
    mask = np.isin(X_train[current_indices, pivot_idx], left_categories)
    return _apply_mask(current_indices, mask)

def _apply_mask(current_indices, mask):
    left_indices, right_indices = current_indices.copy(), current_indices.copy()
    left_indices[current_indices] = mask
    right_indices[current_indices] = ~mask
    return left_indices, right_indices
