from __future__ import annotations

import numpy as np

from .bcSplit import Split

class Tree:
    # allocate memory for attributes, this reduces memory usage
    __slots__ = ("node", "left", "right", "obs_ids", "f_node", "pivot",
                 "pivot_categorical", "intercept_left", "slope_left",
                 "intercept_right", "slope_right", "interval", "wrss",
                 "depth", "model_depth")
    """
    We use a tree object to save the PILOT model.
    Attributes:
    -----------
    node: str,
        type of the regression model
        'lin', 'blin', 'pcon', 'plin' or 'END' to denote the end of the tree
    left: Tree,
        left child node
    right: Tree,
        right child node
    obs_ids: ndarray,
        1D int array indicating the observations in the node
    pivot: tuple,
        a tuple to indicate where we performed a split. The first
        coordinate is the feature_id and the second one is
        the pivot.
    intercept_left: float,
        the intercept of the left child node.
    slope_left: float,
        the slope of the left child node.
    intercept_right: float,
        the intercept of the right child node.
    slope_right: float,
        the slope of the right child node.
    f_node: ndarray,
        a 1D float array which gives the predictions in the node.
    wrss: float,
        the weighted rss in the current node.
    depth: int,
        the depth of the current node/subtree
    model_depth: int,
        the depth of the current node/subtree including linear nodes
    interval: ndarray,
        1D float array for the range of the selected predictor in the training data
    pivot_categorical: ndarry,
        1D int array. Indicating the levels in the left node
        if the selected predictor is categorical
    """
    def __init__(
        self,
        split: Split = Split(),
        f_node=None,
        depth=None,
        model_depth=None,
        obs_ids=None,
    ) -> None:
        """
        Here we input the tree attributes.
        parameters:
        ----------
        split: Split,
            split object containing the split information
        f_node: ndarray,
            a 1D float array which gives the predictions in the node
        depth: int,
            the depth of the current node/subtree
        model_depth: int
            the depth of the current node/subtree including linear nodes
        obs_ids: ndarray,
            1D int array indicating the observations in the node
        """
        self.node = split.node
        self.left = None  # go left by default if node is 'lin'
        self.right = None
        self.obs_ids = obs_ids
        self.f_node = f_node
        self.pivot = (split.feature, split.pivot)
        self.pivot_categorical = split.pivot_categorical
        self.intercept_left = split.intercept_left
        self.slope_left = split.slope_left
        self.intercept_right = split.intercept_right
        self.slope_right = split.slope_right
        self.interval = split.interval
        self.wrss = split.wrss
        self.depth = depth
        self.model_depth = model_depth

    def count_node_types(self, depth=None) -> dict[str, int]:
        """
        count the number of models selected in the tree
        parameters:
        -----------
        depth: int, default = None.
            If specified count the number of models until the
            specified depth.
        """
        node_counts = {
            "con": 0,
            "lin": 0,
            "tlin": 0,
            "blin": 0,
            "pcon": 0,
            "plin": 0,
            "pconc": 0
            }
        if self.node == "END":
            return node_counts
        elif depth is not None and self.depth == depth + 1:
            # if depth is set, stop at the specified depth
            return node_counts
        nodes_l = self.left.count_node_types(depth) if self.left is not None else node_counts
        nodes_l[self.node] += 1
        if self.node in ["plin", "pcon", "pconc", "blin"]:
            nodes_r = self.right.count_node_types(depth) if self.right is not None else node_counts
            for k in nodes_l.keys():
                nodes_l[k] += nodes_r[k]
        return nodes_l

    def get_max_depth(self) -> int:
        depth = self.depth
        left = self.left
        right = self.right
        if left is not None and left.node != "END":
            depth = left.get_max_depth()
        if right is not None and right.node != "END":
            depth = max(depth, right.get_max_depth())
        return depth


def _get_child_data(training_data, model_tree):
    left_data = training_data.copy()
    right_data = training_data.copy()
    if model_tree.node == "pconc":
        left_data = training_data[
            np.isin(training_data[:, model_tree.pivot[0]], model_tree.pivot_categorical)
        ]
        right_data = training_data[
            ~np.isin(training_data[:, model_tree.pivot[0]], model_tree.pivot_categorical)
        ]

    elif model_tree.node in ["plin", "pcon", "blin"]:
        left_data = training_data[training_data[:, model_tree.pivot[0]] <= model_tree.pivot[1]]
        right_data = training_data[training_data[:, model_tree.pivot[0]] > model_tree.pivot[1]]

    return left_data, right_data
