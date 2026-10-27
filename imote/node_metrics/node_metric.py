from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from imote.nodes.base_node import BaseNode
from imote.nodes.split_node import SplitNode

NODE_METRICS_REGISTRY: dict[str, "type[BaseNodeMetric]"] = {}
"""Maps metric names to metric classes, populated via register_metric()."""

def register_metric(name: str):
    """Class decorator that registers a node metric under a given name.

    Args:
        name: Name to register the metric under.

    Returns:
        A decorator that registers the decorated class and returns it
        unchanged.
    """
    def decorator(metric_cls: "type[BaseNodeMetric]") -> "type[BaseNodeMetric]":
        NODE_METRICS_REGISTRY[name] = metric_cls
        return metric_cls
    return decorator

class BaseNodeMetric(ABC):
    """Base class for a single metric computed on a tree node.

    Attributes:
        higher_is_better: True or False if a higher value means a better metric, None if the metric has no better side.
        value_range: (low, high) of the values a metric can have.
    """

    higher_is_better: bool | None = None
    value_range: tuple[float, float] | None = None

    @staticmethod
    @abstractmethod
    def compute(node: BaseNode, y_train: np.ndarray) -> object:
        """Compute the raw value of this metric for a node.

        Args:
            node: The node to compute the metric for.
            y_train: Full training target array.
        Returns:
            The raw computed value (int, float, str, ...).
        """
        raise NotImplementedError

    @staticmethod
    def format(value: object) -> str:
        """Format a computed value for display."""

        if isinstance(value, (bool, np.bool_)):
            return str(value)

        if isinstance(value, (int, np.integer)):
            return str(int(value))

        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return str(value)
            if value == 0:
                return "0"
            if float(value).is_integer():
                return str(int(value))
            return f"{value:.6g}"

        return str(value)

    @classmethod
    def run(cls, node: BaseNode, y_train: np.ndarray) -> str:
        """Compute and format the metric in one step."""
        return cls.format(cls.compute(node, y_train))


# ---------------------------------------------------------------------------
# Generic metrics (apply to any BaseNode)
# ---------------------------------------------------------------------------

@register_metric("ID")
class IdMetric(BaseNodeMetric):
    """Node identifier."""

    @staticmethod
    def compute(node, y_train):
        return node.id

@register_metric("# Samples")
class NSamplesMetric(BaseNodeMetric):
    """Number of samples reaching this node."""

    @staticmethod
    def compute(node, y_train):
        return int(np.sum(node.indices))

@register_metric("RSS")
class RSSMetric(BaseNodeMetric):
    """Residual sum of squares of the node's own residuals (y_res)."""

    higher_is_better = False

    @staticmethod
    def compute(node, y_train):
        return float(np.sum(node.y_res ** 2))

@register_metric("MAE")
class MAEMetric(BaseNodeMetric):
    """Mean absolute error between true and predicted values at this node."""

    higher_is_better = False

    @staticmethod
    def compute(node, y_train):
        return float(np.mean(np.abs(node.y_res)))

@register_metric("MSE")
class MSEMetric(BaseNodeMetric):
    """Mean squared error between true and predicted values at this node."""

    higher_is_better = False

    @staticmethod
    def compute(node, y_train):
        return float(np.mean((node.y_res) ** 2))

@register_metric("RMSE")
class RMSEMetric(BaseNodeMetric):
    """Root mean squared error between true and predicted values at this node."""

    higher_is_better = False

    @staticmethod
    def compute(node, y_train):
        return float(np.sqrt(np.mean((node.y_res) ** 2)))

@register_metric("R2")
class R2Metric(BaseNodeMetric):
    """Coefficient of determination (R^2) at this node."""

    higher_is_better = True
    value_range = (0.0, 1.0)

    @staticmethod
    def compute(node, y_train):
        y_true = y_train[node.indices]
        ss_res = np.sum((node.y_res) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return float("nan")
        return float(1 - ss_res / ss_tot)

@register_metric("Mean Residual")
class MeanResidualMetric(BaseNodeMetric):
    """Mean of the node's own residuals (y_res)."""

    @staticmethod
    def compute(node, y_train):
        return float(np.mean(node.y_res))

@register_metric("Std Residual")
class StdResidualMetric(BaseNodeMetric):
    """Standard deviation of the node's own residuals (y_res)."""

    higher_is_better = False

    @staticmethod
    def compute(node, y_train):
        return float(np.std(node.y_res))

# ---------------------------------------------------------------------------
# Split node specific metrics
# ---------------------------------------------------------------------------

@register_metric("Split Balance")
class SplitBalanceMetric(BaseNodeMetric):
    """Fraction of samples routed to the right child (0.5 = perfectly balanced)."""

    value_range = (0.0, 1.0)

    @staticmethod
    def compute(node, y_train):
        if isinstance(node, SplitNode):
            n_right = np.sum(node.right_child.indices)
            n_total = np.sum(node.indices)
            return n_right / n_total
        return "-"