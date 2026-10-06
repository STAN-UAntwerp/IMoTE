from abc import ABC, abstractmethod
import numpy as np
from typing import List

from imote.nodes.base_node import BaseNode
from imote.nodes.node_model import SimpleLinearNodeModel

class InternalNode(BaseNode, ABC):
    """Abstract base for all internal nodes.

    Attributes:
        pivot_idx: Index of the feature this node acts on.
    """

    pivot_idx: int

    def __init__(self, indices: np.ndarray, y_res: np.ndarray, rss: float, pivot_idx: int):
        """
        Args:
            indices: Row indices of the samples belonging to this node.
            y_res: Residuals of the target variable at this node.
            rss: Residual sum of squares at this node.
            pivot_idx: Index of the feature this node acts on.
        """
        super().__init__(indices, y_res, rss)
        self.pivot_idx = pivot_idx

    def get_minimal_label(self) -> str:
        """Returns a short display label for this node.

        Returns:
            String of minimal label, used when space is limited.
        """
        return f"X{self.pivot_idx}"

    @abstractmethod
    def set_children(self, children: List[BaseNode]):
        """Replaces this node's children.

        Args:
            children: New child node(s), in the same order as
                returned by get_children().
        """
        pass

@BaseNode.register
class LinearNode(InternalNode):
    """LIN: Single child node that applies a linear model..

    Attributes:
        linear_model: Single-feature linear model applied at this node.
        child: The node's single child.
    """

    linear_model: SimpleLinearNodeModel
    child: BaseNode

    def __init__(self, indices: np.ndarray, y_res: np.ndarray, rss: float, pivot_idx: int,
                 linear_model: SimpleLinearNodeModel, child: BaseNode):
        """
        Args:
            indices: Row indices of the samples associated with this node.
            y_res: Residuals of the target variable at this node.
            rss: Residual sum of squares at this node.
            pivot_idx: Index of the feature the linear term is based on.
            linear_model: One dimensional linear model applied at this node.
            child: The node's single child.
        """
        super().__init__(indices, y_res, rss, pivot_idx)
        self.linear_model = linear_model
        self.child = child

    def get_children(self) -> List[BaseNode]:
        """Returns this node's single child.

        Returns:
            List containing the child node.
        """
        return [self.child]

    def set_children(self, children: List[BaseNode]):
        """Replaces this node's child.

        Args:
            children: List containing exactly one new child node.
        """
        self.child = children[0]

    def get_label(self) -> str:
        """Returns the full display label for this node.

        Returns:
            Label of the form "LIN\\nX<pivot_idx>".
        """
        return f"LIN\nX{self.pivot_idx}"

    def to_dict(self) -> dict:
        """Serializes the node to a dictionary.

        Returns:
            Dictionary with base attributes plus pivot_idx, linear_model,
            and the child's class name and serialized form.
        """
        node_dict = super().to_dict()
        node_dict.update({
            "pivot_idx": self.pivot_idx,
            "linear_model": self.linear_model.to_dict(),
            "child_class": self.child.__class__.__name__,
            "child_dict": self.child.to_dict()
        })
        return node_dict

    @classmethod
    def from_dict(cls, dic: dict) -> "LinearNode":
        """Reconstructs a LinearNode instance, including its child.

        Args:
            dic: Dictionary previously produced by to_dict().

        Returns:
            A new LinearNode instance with its child restored.
        """
        child_class = cls.class_registry[dic["child_class"]]

        obj = super().from_dict(dic)
        obj.pivot_idx = dic["pivot_idx"]
        obj.linear_model = SimpleLinearNodeModel.from_dict(dic["linear_model"])
        obj.child = child_class.from_dict(dic["child_dict"])
        return obj