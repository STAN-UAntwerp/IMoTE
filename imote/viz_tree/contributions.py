import numpy as np

from imote.viz_tree.viz_tree import VizTree
from imote.nodes.collapsed_node import CollapsedNode
from imote.nodes.internal_node import InternalNode, LinearNode
from imote.nodes.split_node import PconNode, PconcNode, SplitNode, BlinNode, PlinNode
from imote.nodes.leaf_node import LeafNode
from imote.nodes.node_model import LinearNodeModel

########################################
#
# Contributions used in predsplot.py with a leaf node vs this file:
#
# When predsplot.py (with intercept==None) is run with the information of a leaf node: contributions are the added
# prediction on top of the average prediction for that leaf node (predplot.py has no info about the trees structure)
#
# In this file: contributions are the added prediction on top of the average prediction using the data of the
# node where the model (lin, plin, pcon,...) is fitted.
#
# With this adjustment to contributions compared to predsplot.py the total contribution also contains the split
# contributions and is not limited to linear contributions.
#
# Remark: The data used when a model is fitted can differ from the data in a leaf node, so the linear contributions in
# this file and the contributions in predsplot.py can also differ (slightly).
#
########################################


def get_split_and_lin_contributions_test(viz_tree: VizTree, x_test: np.ndarray):
    split_contributions = np.zeros_like(x_test, dtype=float)
    lin_contributions = np.zeros_like(x_test, dtype=float)

    X = viz_tree.X_train
    node = viz_tree.root_node
    while not isinstance(node, LeafNode):
        # Calculate contributions
        if isinstance(node, CollapsedNode):
            raise ValueError("CollapsedNode cannot be used when calculating contributions")
        if isinstance(node, InternalNode):
            pivot = node.pivot_idx
            if isinstance(node, LinearNode):
                linear_predictions = node.linear_model.predict(X[node.child.indices, pivot])
                linear_prediction_x_test = node.linear_model.predict(np.array([x_test[pivot]]))[0]
                lin_contributions[pivot] += linear_prediction_x_test - np.mean(linear_predictions)
            else:
                if isinstance(node, (PconcNode, PconNode)):
                    avg_pred_left = node.left_model.predict(np.array([0]))[0]
                    avg_pred_right = node.right_model.predict(np.array([0]))[0]
                elif isinstance(node, (BlinNode, PlinNode)):
                    linear_predictions_left = node.left_model.predict(X[node.left_child.indices, pivot])
                    linear_predictions_right = node.right_model.predict(X[node.right_child.indices, pivot])
                    avg_pred_left = np.mean(linear_predictions_left)
                    avg_pred_right = np.mean(linear_predictions_right)
                    linear_prediction_x_test_left = node.left_model.predict(np.array([x_test[pivot]]))[0]
                    linear_prediction_x_test_right = node.right_model.predict(np.array([x_test[pivot]]))[0]
                elif isinstance(node, SplitNode):
                    split_contributions, lin_contributions, _ = _calculate_split_and_lin_contributions_test2_recursive(
                        viz_tree, viz_tree.root_node, x_test)
                    return split_contributions, lin_contributions
                else:
                    raise ValueError(f"Node of class {node.__class__.__name__} is not implemented to calculate contributions")

                avg_pred = (avg_pred_left*np.sum(node.left_child.indices) + avg_pred_right*np.sum(node.right_child.indices))/np.sum(node.indices)

        # Find next node in path
        if isinstance(node, LinearNode):
            node = node.child
        elif node.goes_left(x_test[pivot]):
            split_contributions[pivot] += avg_pred_left - avg_pred
            if not isinstance(node, PconNode):
                lin_contributions[pivot] += linear_prediction_x_test_left - avg_pred_left
            node = node.left_child
        else:
            split_contributions[pivot] += avg_pred_right - avg_pred
            if not isinstance(node, PconNode):
                lin_contributions[pivot] += linear_prediction_x_test_right - avg_pred_right
            node = node.right_child

    return split_contributions, lin_contributions

# For non-pilot trees
def _calculate_split_and_lin_contributions_test2_recursive(viz_tree: VizTree, node, x_test: np.ndarray):
    if isinstance(node, LeafNode):
        node_model = node.node_model
        X_node = viz_tree.X_train[node.indices, :]
        avg_pred = np.mean(node_model.predict(X_node))

        split_contributions = np.zeros_like(x_test, dtype=float)
        lin_contributions = np.zeros_like(x_test, dtype=float)
        if isinstance(node_model, LinearNodeModel):
            lin_contributions += node_model.coefficients * (x_test - np.mean(X_node, axis=0))
        return split_contributions, lin_contributions, avg_pred
    elif isinstance(node, SplitNode):
        split_contributions_left, lin_contributions_left, avg_pred_left = (
            _calculate_split_and_lin_contributions_test2_recursive(viz_tree, node.left_child, x_test))
        split_contributions_right, lin_contributions_right, avg_pred_right = (
            _calculate_split_and_lin_contributions_test2_recursive(viz_tree, node.right_child, x_test))
        pivot = node.pivot_idx
        avg_pred = (avg_pred_left * np.sum(node.left_child.indices) +
                    avg_pred_right * np.sum(node.right_child.indices)) / np.sum(node.indices)
        if node.goes_left(x_test[pivot]):
            split_contributions = split_contributions_left
            split_contributions[pivot] += avg_pred_left - avg_pred
            lin_contributions = lin_contributions_left
        else:
            split_contributions = split_contributions_right
            split_contributions[pivot] += avg_pred_right - avg_pred
            lin_contributions = lin_contributions_right

        return split_contributions, lin_contributions, avg_pred
    else:
        raise NotImplementedError(
            f"Node of class {node.__class__.__name__} is not implemented to calculate contributions")
