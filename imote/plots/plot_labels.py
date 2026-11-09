from imote.config import MINIMAL_LABEL_NAME_LENGTH
from imote.node_metrics.node_labels import display_names

MAX_FEATURE_LABEL_LENGTH = 8


def plot_math_string(text: str) -> str:
    """Escapes dollar signs so matplotlib doesn't read the text as math."""
    return str(text).replace("$", r"\$")


def feature_label(feature_names, idx: int, max_length: int = MAX_FEATURE_LABEL_LENGTH) -> str:
    """Returns a plot label for feature idx, shortened to max_length characters."""
    if feature_names is None:
        return "$X_{" + f"{idx}" + "}$"
    name = str(feature_names[idx])
    if len(name) > max_length:
        name = name[:max_length - 1] + "…"
    return plot_math_string(name)


def feature_labels(feature_names, n_features: int) -> list[str]:
    """Returns the plot labels of all features, shortened the same way as the node labels in minimal view."""
    if feature_names is None:
        return [feature_label(None, i) for i in range(n_features)]
    short_names = display_names([str(name) for name in feature_names], n_features, MINIMAL_LABEL_NAME_LENGTH)
    return [plot_math_string(name) for name in short_names]
