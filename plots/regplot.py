import numpy as np
import matplotlib.pyplot as plt
from nodes.internal_node import InternalNode, LinearNode
from nodes.split_node import PconcNode, SplitNode, PconNode, SplitCNode, PlinNode, BlinNode


def make_regression_plot(node: InternalNode, viz_tree_X, directory_regplot_file, fig_size, feature_colors, feature_names, highlight_x=None):
    X = viz_tree_X[node.indices,:]
    
    fig = plt.figure(figsize=fig_size, layout="constrained")
    plt.gca().ticklabel_format(scilimits=[-3, 4])
    w = np.ones(len(node.y_res))
    feature_idx = node.pivot_idx
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor(feature_colors[feature_idx])
    if feature_names is None:
        feature_label = "$X_{" + f"{feature_idx}" + "}$"
    else:
        feature_label = feature_names[feature_idx]
    min_x = X[:, feature_idx].min()
    max_x = X[:, feature_idx].max()
    scaled_weights = (w.flatten() - np.mean(w)) * 100 + 10
    plt.scatter(X[:, feature_idx], node.y_res, s=scaled_weights, color='slategrey')
    if highlight_x is not None:
        matches = np.all(X == highlight_x, axis=1)
        idx_point = np.argmax(matches) if np.any(matches) else None
        if idx_point is None:
            plt.axvline(x=highlight_x[feature_idx], linestyle='--', color='r')
        else:
            plt.scatter(X[idx_point, feature_idx], node.y_res[idx_point], s=60 + scaled_weights[idx_point],
                        facecolors='r', marker='*')
    if isinstance(node, LinearNode):
        x = np.array([min_x, max_x])
        y = node.linear_model.predict(x)
        plt.plot(x, y, color=feature_colors[feature_idx], linewidth=3)
        plt.title(f"LIN - Feature: {feature_label}")

    elif isinstance(node, (PconcNode,SplitCNode)):
        left_child_values = node.pivot_value
        possible_values = np.unique(X[:, feature_idx])
        for i in range(len(possible_values)):
            value = possible_values[i]
            if i == 0:
                left_len = right_len = (possible_values[i+1] - value)/2
            elif i == len(possible_values) - 1:
                left_len = right_len = (value - possible_values[i-1])/2
            else:
                left_len = (value - possible_values[i-1])/2
                right_len = (possible_values[i+1] - value)/2
            if value in left_child_values:
                y = node.left_model.predict(np.array([value]))
            else:
                y = node.right_model.predict(np.array([value]))
            plt.plot([value-left_len, value+right_len], [y, y], color=feature_colors[feature_idx], linewidth=3)
        self_name = str(node.__class__.__name__)
        plt.title(f"{self_name} - Feature: {feature_label}")

    elif isinstance(node, (PconNode, PlinNode, BlinNode)):
        pivot = node.pivot_value
        x1 = np.array([min_x, pivot]).reshape(2, -1)
        x2 = np.array([pivot, max_x]).reshape(2, -1)
        y1 = node.left_model.predict(x1)
        y2 = node.right_model.predict(x2)
        plt.plot(x1, y1, x2, y2, color=feature_colors[feature_idx], linewidth=3)
        self_name = str(node.__class__.__name__)
        plt.title(f"{self_name} - Feature: {feature_label} - Pivot: {pivot:.3g}")

    elif isinstance(node, SplitNode):
        pivot = node.pivot_value
        plt.axvline(x=pivot, color='k')
        self_name = str(node.__class__.__name__)
        plt.title(f"{self_name} - Feature: {feature_label} - Pivot: {pivot:.3g}")

    plt.xlabel(feature_label)
    if isinstance(node, (SplitNode, SplitCNode)):
        y_label = "y"
    else:
        y_label = "y" if node.id == 0 else "Residuals"
    plt.ylabel(y_label)
    fig.savefig(directory_regplot_file)
    plt.close(fig)
