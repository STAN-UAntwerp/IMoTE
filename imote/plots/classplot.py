import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from imote.nodes.internal_node import InternalNode, LinearNode
from imote.nodes.split_node import SplitNode
from imote.plots.plot_labels import feature_label as make_feature_label

MAX_DISCRETE_VALUES = 20
"""Features with at most this many unique values in the node get one bar per value."""
MAX_BINS = 30
LEFT_SPAN_COLOR = "k"
LEFT_SPAN_ALPHA = 0.12


def _class_label(c) -> str:
    if isinstance(c, (float, np.floating)) and float(c).is_integer():
        c = int(c)
    return f"Class {c}"


def _bin_edges(x: np.ndarray, node: InternalNode) -> tuple[np.ndarray, bool]:
    """Returns the bin edges and whether the feature is shown as discrete values.

    For a numeric split the pivot is a bin edge and both sides use the same bin width.
    """
    values = np.unique(x)
    if len(values) <= MAX_DISCRETE_VALUES:
        return np.unique(np.r_[values - 0.5, values + 0.5]), True

    if isinstance(node, SplitNode) and not node.categorical and values[0] < node.pivot_value < values[-1]:
        pivot = node.pivot_value
        auto_edges = np.histogram_bin_edges(x, bins="auto")
        width = max(auto_edges[1] - auto_edges[0], (values[-1] - values[0]) / MAX_BINS)
        n_left = int(np.ceil((pivot - values[0]) / width))
        n_right = int(np.ceil((values[-1] - pivot) / width))
        return pivot + width * np.arange(-n_left, n_right + 1), False

    edges = np.histogram_bin_edges(x, bins="auto")
    if len(edges) > MAX_BINS + 1:
        edges = np.linspace(values[0], values[-1], MAX_BINS + 1)
    return edges, False


def _pivot_line_position(x: np.ndarray, pivot: float, discrete: bool) -> float:
    if not discrete:
        return pivot
    values = np.unique(x)
    left, right = values[values <= pivot], values[values > pivot]
    if len(left) == 0 or len(right) == 0:
        return pivot
    return (left[-1] + right[0]) / 2


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
    feature_label = make_feature_label(feature_names, feature_idx, max_length=40)

    edges, discrete = _bin_edges(x, node)
    plt.hist([x[y == c] for c in classes], bins=edges, stacked=True,
             color=[class_colors[c] for c in classes], label=[_class_label(c) for c in classes],
             rwidth=0.85 if discrete else None, edgecolor="white", linewidth=0.3)
    if discrete:
        plt.xticks(np.unique(x) if len(np.unique(x)) <= 10 else None)

    node_name = node.__class__.__name__
    left_legend_handle = None
    if isinstance(node, LinearNode):
        plt.title(f"LIN - Feature: {feature_label}")
    elif isinstance(node, SplitNode) and node.categorical:
        left_values = ", ".join(f"{v:g}" for v in node.pivot_value)
        values = np.unique(x)
        for i, value in enumerate(values):
            if value in node.pivot_value:
                lo, hi = (value - 0.5, value + 0.5)
                plt.axvspan(lo, hi, color=LEFT_SPAN_COLOR, alpha=LEFT_SPAN_ALPHA, linewidth=0)
        left_legend_handle = Patch(facecolor=LEFT_SPAN_COLOR, alpha=LEFT_SPAN_ALPHA,
                                   label=f"{{{left_values}}} go left")
        plt.title(f"{node_name} - Feature: {feature_label} - Left: {{{left_values}}}")
    elif isinstance(node, SplitNode):
        plt.axvline(x=_pivot_line_position(x, node.pivot_value, discrete), color="k", linewidth=1.5)
        plt.title(f"{node_name} - Feature: {feature_label} - Pivot: {node.pivot_value:.3g}")
    else:
        plt.title(f"{node_name} - Feature: {feature_label}")

    if highlight_x is not None:
        plt.axvline(x=highlight_x[feature_idx], linestyle="--", color="r")

    plt.xlabel(feature_label)
    plt.ylabel("Count")
    ax = plt.gca()
    class_legend = ax.legend(fontsize="small", ncols=1 if len(classes) <= 5 else 2, loc="upper right")
    if left_legend_handle is not None:
        ax.add_artist(class_legend)
        ax.legend(handles=[left_legend_handle], fontsize="small", loc="upper left")
    fig.savefig(directory_plot_file)
    plt.close(fig)
