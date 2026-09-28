import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shap.plots import colors
from datetime import datetime
import os
from subprocess import run

from viz_tree.viz_tree import VizTree
from nodes.collapsed_node import CollapsedNode
from nodes.internal_node import InternalNode, LinearNode
from nodes.split_node import PconNode, PconcNode, SplitNode, BlinNode, PlinNode, SplitCNode
from nodes.leaf_node import LeafNode
from nodes.node_model import LinearNodeModel
from plots.beeswarm import beeswarm
import matplotlib.pyplot as plt

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
        elif isinstance(node, PconcNode):
            if x_test[pivot] in node.pivot_value:
                node = node.left_child
                split_contributions[pivot] += avg_pred_left - avg_pred
            else:
                node = node.right_child
                split_contributions[pivot] += avg_pred_right - avg_pred
        else:
            if x_test[pivot] > node.pivot_value:
                split_contributions[pivot] += avg_pred_right - avg_pred
                if not isinstance(node, PconNode):
                    lin_contributions[pivot] += linear_prediction_x_test_right - avg_pred_right
                node = node.right_child
            else:
                split_contributions[pivot] += avg_pred_left - avg_pred
                if not isinstance(node, PconNode):
                    lin_contributions[pivot] += linear_prediction_x_test_left - avg_pred_left
                node = node.left_child

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
        if isinstance(node, SplitCNode):
            if x_test[pivot] in node.pivot_value:
                split_contributions = split_contributions_left
                split_contributions[pivot] += avg_pred_left - avg_pred
                lin_contributions = lin_contributions_left
            else:
                split_contributions = split_contributions_right
                split_contributions[pivot] += avg_pred_right - avg_pred
                lin_contributions = lin_contributions_right
        else:
            if x_test[node.pivot_idx] > node.pivot_value:
                split_contributions = split_contributions_right
                split_contributions[pivot] += avg_pred_right - avg_pred
                lin_contributions = lin_contributions_right
            else:
                split_contributions = split_contributions_left
                split_contributions[pivot] += avg_pred_left - avg_pred
                lin_contributions = lin_contributions_left

        return split_contributions, lin_contributions, avg_pred
    else:
        raise NotImplementedError(
            f"Node of class {node.__class__.__name__} is not implemented to calculate contributions")

def scatter_contributions(df_X, df_contributions, feature, color_feature = None):
    if color_feature is None:
        corr = df_contributions.corr()
        color_feature = corr[feature].nlargest(2).index[1]

    x_color = df_X[color_feature].to_numpy()
    median = np.nanmedian(x_color)
    mad = 1.4826 * np.nanmedian(np.abs(x_color - median))
    vmin = median - 3 * mad
    vmax = median + 3 * mad


    fig, ax = plt.subplots(figsize=(8, 6))

    # scatter
    sc = ax.scatter(
        df_X[feature],
        df_contributions[feature],
        c=df_X[color_feature].clip(vmin, vmax),
        cmap=colors.red_blue,
        s=10,
        marker='.',
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f"{feature} contribution")
    ax.set_xlabel(feature)
    ax.set_ylabel("Contribution")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color_feature)

    # histogram inset at bottom
    ax_hist = inset_axes(
        ax,
        width="100%",
        height="18%",
        loc="lower left",
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    ax_hist.hist(
        df_X[feature],
        bins=50,
        color="gray",
        alpha=0.3
    )

    ax_hist.set_yticks([])
    ax_hist.set_xticks([])
    ax_hist.patch.set_alpha(0)

    for spine in ax_hist.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()

def beeswarm_wrap(df_contributions, X, y_hat, directory):
    id_results = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
    file_directory = os.path.join(directory, f"beeswarm_{id_results}.pdf")
    feature_names = df_contributions.columns.tolist()
    beeswarm(np.array(df_contributions),
             X,
             y_hat,
             n_max=10,
             fig_size=(10, 6),
             truncate_total_pred=True,
             variable_tick_width=True,
             file_directory=file_directory,
             highlight_x=None,
             staircase=False,
             feature_names=feature_names,
             )
    run(["open", "-a", "Preview", file_directory])