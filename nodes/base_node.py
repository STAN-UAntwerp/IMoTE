from abc import ABC, abstractmethod
import numpy as np
from typing import List, Type, TypeVar

T = TypeVar("T", bound="BaseNode")


class BaseNode(ABC):
    """Abstract base class for all node types in a tree.

    Attributes:
        id: Unique identifier assigned to the node, or None if unset.
        indices: Row indices of the samples associated with this node.
        y_res: Residuals of the target variable at this node.
        rss: Residual sum of squares at this node.
    """
    id: int
    indices: np.ndarray
    y_res: np.ndarray
    rss: float

    def __init__(self, indices: np.ndarray, y_res: np.ndarray, rss: float):
        """
        Args:
            indices: Row indices of the samples belonging to this node.
            y_res: Residuals of the target variable at this node.
            rss: Residual sum of squares at this node.
        """
        self.indices = indices
        self.y_res = y_res
        self.rss = rss

    def set_id(self, id: int):
        """Sets the unique identifier of the node.

        Args:
            id: Identifier to assign.
        """
        self.id = id

    @abstractmethod
    def get_children(self) -> List["BaseNode"]:
        """Returns the list of child nodes of this node.

        Returns:
            List of child BaseNode instances (empty for leaf-like nodes).
        """
        pass

    def get_all_children(self) -> List["BaseNode"]:
        """Returns all descendant nodes of this node, recursively.

        Returns:
            Flat list of all descendant BaseNode instances, in
            depth-first order.
        """
        return self.get_children() + sum([child.get_all_children() for child in self.get_children()], [])

    @abstractmethod
    def get_label(self) -> str:
        """Returns the full display label for this node.

        Returns:
            String of label, shown in a tree visualization.
        """
        pass

    @abstractmethod
    def get_minimal_label(self) -> str:
        """Returns a short display label for this node.

        Returns:
            String of minimal label, used when space is limited.
        """
        pass

    def to_dict(self) -> dict:
        """Serializes the node's base attributes to a dictionary.

        Subclasses should call this and extend the result with their
        own attributes.

        Returns:
            Dictionary with the node's id, rss, indices, and y_res.
        """
        node_dict = {"id": self.id, "rss": self.rss}
        if isinstance(self.indices, list):
            node_dict["indices"] = self.indices
        else:
            node_dict["indices"] = self.indices.tolist()
        if isinstance(self.y_res, list):
            node_dict["y_res"] = self.y_res
        else:
            node_dict["y_res"] = self.y_res.tolist()
        return node_dict

    @classmethod
    def from_dict(cls, node_dict: dict) -> "BaseNode":
        """Reconstructs the node's base attributes from a dictionary.

        Subclasses should call this and populate their own attributes
        on the returned object.

        Args:
            node_dict: Dictionary of node previously produced by to_dict().

        Returns:
            A new instance of cls with base attributes restored.
        """
        obj = cls.__new__(cls)
        obj.id = node_dict["id"]
        obj.indices = np.array(node_dict["indices"])
        obj.y_res = np.array(node_dict["y_res"])
        obj.rss = node_dict["rss"]
        return obj

    class_registry = {}
    """Maps subclass names to subclass objects, for use in from_dict()."""

    @classmethod
    def register(cls, subclass: Type[T]) -> Type[T]:
        """Class decorator that registers a node subclass by name.

        Registration is required so that from_dict() can look up the
        correct class when deserializing a node whose type is only
        known by name (e.g. a child stored as "left_child_class").

        Args:
            subclass: The BaseNode subclass to register.

        Returns:
            The subclass, unchanged (so it can be used as a decorator).
        """
        cls.class_registry[subclass.__name__] = subclass
        return subclass