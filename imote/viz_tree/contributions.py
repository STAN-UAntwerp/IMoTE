import numpy as np
from typing import Tuple

from imote.nodes.base_node import BaseNode
from imote.nodes.collapsed_node import CollapsedNode
from imote.nodes.internal_node import LinearNode
from imote.nodes.split_node import SplitNode
from imote.nodes.leaf_node import LeafNode
from imote.nodes.node_model import LinearNodeModel


def calculate_contributions(root_node: BaseNode, X_train: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the split and linear contributions of each feature for every row of X.

    Args:
        root_node: Root node of the tree.
        X_train: Training feature matrix, used for the average predictions and means of the nodes.
        X: Feature matrix to compute the contributions for, one row per sample.

    Returns:
        Tuple of split and linear contributions, both matrices with the same shape as X.
    """
    split_contributions = np.zeros_like(X, dtype=float)
    lin_contributions = np.zeros_like(X, dtype=float)
    _add_contributions(root_node, X_train, X, np.arange(X.shape[0]), split_contributions, lin_contributions)
    return split_contributions, lin_contributions


def _add_contributions(node: BaseNode, X_train: np.ndarray, X: np.ndarray, rows: np.ndarray,
                       split_contributions: np.ndarray, lin_contributions: np.ndarray) -> float:
    """Adds the contributions of the subtree of node to the given rows of X, recursively.

    Returns:
        Average prediction over the training data in node.
    """
    children = node.get_children()
    if children and isinstance(children[0], CollapsedNode):
        node = children[0].parent  # copy of node that still has its children

    if isinstance(node, LeafNode):
        X_node = X_train[node.indices, :]
        if isinstance(node.node_model, LinearNodeModel):
            lin_contributions[rows, :] += node.node_model.coefficients * (X[rows, :] - np.mean(X_node, axis=0))
        return np.mean(node.node_model.predict(X_node))
    if isinstance(node, LinearNode):
        return _add_contributions(node.child, X_train, X, rows, split_contributions, lin_contributions)
    if isinstance(node, SplitNode):
        mask_left = node.goes_left_mask(X[rows, node.pivot_idx])
        avg_pred_left = _add_contributions(node.left_child, X_train, X, rows[mask_left],
                                           split_contributions, lin_contributions)
        avg_pred_right = _add_contributions(node.right_child, X_train, X, rows[~mask_left],
                                            split_contributions, lin_contributions)
        avg_pred = (avg_pred_left * np.sum(node.left_child.indices) +
                    avg_pred_right * np.sum(node.right_child.indices)) / np.sum(node.indices)
        split_contributions[rows[mask_left], node.pivot_idx] += avg_pred_left - avg_pred
        split_contributions[rows[~mask_left], node.pivot_idx] += avg_pred_right - avg_pred
        return avg_pred
    raise NotImplementedError(f"Node of class {node.__class__.__name__} is not implemented to calculate contributions")
