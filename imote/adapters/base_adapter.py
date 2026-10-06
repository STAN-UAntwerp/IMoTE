from abc import ABC, abstractmethod
import numpy as np
from imote.nodes.base_node import BaseNode

ADAPTERS_REGISTRY: dict[str, type["BaseAdapter"]] = {}
"""Maps adapter names to adapter classes, populated via register_adapter()."""

def register_adapter(name: str):
    """Class decorator that registers an adapter under a given name.

    Args:
        name: Name to register the adapter under.

    Returns:
        A decorator that registers the decorated class and returns it
        unchanged.
    """
    def decorator(adapter_cls):
        ADAPTERS_REGISTRY[name] = adapter_cls
        return adapter_cls
    return decorator

class BaseAdapter(ABC):
    """Abstract base class for adapters that build a VizTree from a model.

    An adapter translates a fitted model from a specific source
    into the BaseNode linked tree structure that VizTree expects.
    """

    @staticmethod
    @abstractmethod
    def build_root_node(X_train: np.ndarray, y_train: np.ndarray, model) -> BaseNode:
        """Builds the BaseNode linked tree representing a fitted model.

        Args:
            X_train: Training feature matrix the model was fit on.
            y_train: Training target values the model was fit on.
            model: The fitted model, in whatever format this adapter
                supports (as returned by load_model()).

        Returns:
            Root BaseNode of the reconstructed linked tree.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_model(model_path: str):
        """Loads a fitted model from disk.

        Args:
            model_path: Path to the serialized model file.

        Returns:
            The loaded model, in the format this adapter's
            build_root_node() expects.
        """
        pass

    @staticmethod
    def predict(X: np.ndarray, model) -> np.ndarray | None:
        """Predicts target values directly from the underlying model, if possible.

        This is optional: an adapter can override it to use the
        original model's own prediction logic (e.g. for speed or
        numerical parity). If not overridden, VizTree instead computes
        predictions by traversing the built BaseNode linked tree.

        Args:
            X: Feature matrix to predict on.
            model: The fitted model, as returned by load_model().

        Returns:
            Array of predicted values, or None if not implemented.
        """
        return None