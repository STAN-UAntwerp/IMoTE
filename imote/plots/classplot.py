import numpy as np
import matplotlib.pyplot as plt
from imote.nodes.internal_node import InternalNode, LinearNode
from imote.nodes.split_node import SplitNode

MAX_DISCRETE_VALUES = 20
"""Features with at most this many unique values in the node get one bar per value."""
MAX_BINS = 30
MAX_BINS_HALF = 20


def _class_label(c) -> str:
    if isinstance(c, (float, np.floating)) and float(c).is_integer():
        c = int(c)
    return f"Class {c}"


def _bin_edges(x: np.ndarray, node: InternalNode) -> tuple[np.ndarray, bool]:
    """Returns the bin edges and whether the feature is shown as discrete values.

    The bins on each side of the pivot are made separately so the pivot is not inside a bin.
    """
    values = np.unique(x)
    if len(values) <= MAX_DISCRETE_VALUES:
        gaps = np.diff(values)
        half = gaps.min() / 2 if len(gaps) else 0.5
        return np.r_[values - half, values[-1] + half], True

    if isinstance(node, SplitNode) and not node.categorical and values[0] < node.pivot_value < values[-1]:
        pivot = node.pivot_value
        edges_left = np.histogram_bin_edges(x[x >= pivot], bins="auto")
        if len(edges_left) > MAX_BINS_HALF + 1:
            edges_left = np.linspace(edges_left[0], edges_left[-1], MAX_BINS_HALF + 1)

        edges_right = np.histogram_bin_edges(x[x <= pivot], bins="auto")
        if len(edges_right) > MAX_BINS_HALF + 1:
            edges_right = np.linspace(edges_right[0], edges_right[-1], MAX_BINS_HALF + 1)

        return np.r_[edges_left, edges_right[1:]], False

    edges = np.histogram_bin_edges(x, bins="auto")
    if len(edges) > MAX_BINS + 1:
        edges = np.linspace(values[0], values[-1], MAX_BINS + 1)
    return edges, False


def make_classification_plot(node: InternalNode, viz_tree_X, viz_tree_y, directory_plot_file, fig_size,
                             feature_colors, feature_names, highlight_x=None):
    """Stacked histogram of the node's pivot feature, one color per class."""
    X = viz_tree_X[node.indices, :]
    y = viz_tree_y[node.indices]
    feature_idx = node.pivot_idx
    x = X[:, feature_idx]
    classes = np.unique(viz_tree_y)
    cmap = plt.colormaps["tab10"] if len(classes) <= 10 else plt.colormaps["tab20"].resampled(len(classes))
    class_colors = {class_: cmap(i % cmap.N) for i, class_ in enumerate(classes)}

    fig = plt.figure(figsize=fig_size, layout="constrained")
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor(feature_colors[feature_idx])
    if feature_names is None:
        feature_label = "$X_{" + f"{feature_idx}" + "}$"
    else:
        feature_label = feature_names[feature_idx]

    edges, discrete = _bin_edges(x, node)
    plt.hist([x[y == c] for c in classes], bins=edges, stacked=True,
             color=[class_colors[c] for c in classes], label=[_class_label(c) for c in classes],
             rwidth=0.85 if discrete else None, edgecolor="white", linewidth=0.3)
    if discrete:
        plt.xticks(np.unique(x) if len(np.unique(x)) <= 10 else None)

    node_name = node.__class__.__name__
    if isinstance(node, LinearNode):
        plt.title(f"LIN - Feature: {feature_label}")
    elif isinstance(node, SplitNode) and node.categorical:
        # mark the categories that go to the left child
        for value in node.pivot_value:
            plt.axvspan(value - 0.5 * (edges[1] - edges[0]), value + 0.5 * (edges[1] - edges[0]),
                        color="k", alpha=0.08, linewidth=0)
        left_values = ", ".join(f"{v:g}" for v in node.pivot_value)
        plt.title(f"{node_name} - Feature: {feature_label} - Left: {{{left_values}}}")
    elif isinstance(node, SplitNode):
        plt.axvline(x=node.pivot_value, color="k", linewidth=1.5)
        plt.title(f"{node_name} - Feature: {feature_label} - Pivot: {node.pivot_value:.3g}")
    else:
        plt.title(f"{node_name} - Feature: {feature_label}")

    if highlight_x is not None:
        plt.axvline(x=highlight_x[feature_idx], linestyle="--", color="r")

    plt.xlabel(feature_label)
    plt.ylabel("Count")
    plt.legend(fontsize="small", ncols=1 if len(classes) <= 5 else 2)
    fig.savefig(directory_plot_file)
    plt.close(fig)
