"""IMoTE: Interactive Model Tree Explorer."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("imote")  # set from the git tag by setuptools-scm at install time
except PackageNotFoundError:  # running from a source tree that was never pip installed
    __version__ = "0+unknown"

from imote.adapters import M5Adapter, PartyKitAdapter, PilotAdapter
from imote.adapters.base_adapter import ADAPTERS_REGISTRY, BaseAdapter, register_adapter
from imote.dataset.dataset import Dataset
from imote.node_metrics.node_metric import NODE_METRICS_REGISTRY, BaseNodeMetric, register_metric
from imote.viz_tree.viz_tree import VizTree

__all__ = [
    "VizTree",
    "Dataset",
    "BaseAdapter",
    "register_adapter",
    "ADAPTERS_REGISTRY",
    "PilotAdapter",
    "M5Adapter",
    "PartyKitAdapter",
    "BaseNodeMetric",
    "register_metric",
    "NODE_METRICS_REGISTRY",
]
