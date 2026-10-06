from typing import List, Union
import numpy as np
from imote.nodes.base_node import BaseNode
from imote.nodes.node_model import (NodeModel, NoneNodeModel, ConstantNodeModel, SimpleLinearNodeModel,
                              node_model_from_dict)
from imote.nodes.internal_node import InternalNode


@BaseNode.register
class SplitNode(InternalNode):
    """Split node with a left and right child.

    The plain SplitNode only splits the data and applies no model, so both
    side models are NoneNodeModel. Subclasses change the expected model type
    (model_type) and/or make the split categorical (categorical).

    Attributes:
        pivot_value: Threshold for a numeric split (left if x <= pivot_value),
            or the list of categories routed to the left child for a
            categorical split.
        left_child: Child node for samples on the left of the split.
        right_child: Child node for samples on the right of the split.
        left_model: Model applied to samples on the left of the split.
        right_model: Model applied to samples on the right of the split.
    """

    model_type = NoneNodeModel
    """Model class both side models must be an instance of."""
    categorical = False
    """Whether pivot_value is a list of left categories instead of a threshold."""

    pivot_value: Union[float, list]
    left_child: BaseNode
    right_child: BaseNode
    left_model: NodeModel
    right_model: NodeModel

    def __init__(self, indices: np.ndarray, y_res: np.ndarray, rss: float,
                 pivot_idx: int, pivot_value: Union[float, list],
                 left_child: BaseNode, right_child: BaseNode,
                 left_model: NodeModel, right_model: NodeModel):
        """
        Args:
            indices: Row indices of the samples belonging to this node.
            y_res: Residuals of the target variable at this node.
            rss: Residual sum of squares at this node.
            pivot_idx: Index of the feature the split is based on.
            pivot_value: Threshold value, or list of left categories for a
                categorical split.
            left_child: Child node for samples on the left of the split.
            right_child: Child node for samples on the right of the split.
            left_model: Model applied to samples on the left of the split.
            right_model: Model applied to samples on the right of the split.

        Raises:
            TypeError: If a side model is not an instance of model_type, or
                pivot_value doesn't match the kind of split.
        """
        super().__init__(indices, y_res, rss, pivot_idx)
        self.pivot_value = self._check_pivot_value(pivot_value)
        self.left_child = left_child
        self.right_child = right_child
        self.left_model = left_model
        self.right_model = right_model
        self._check_models()

    def _check_pivot_value(self, pivot_value):
        if self.categorical:
            if not isinstance(pivot_value, (list, tuple, np.ndarray)):
                raise TypeError(f"{type(self).__name__} expects a list of categories as pivot_value, "
                                f"got {type(pivot_value).__name__}")
            return list(pivot_value)
        if np.ndim(pivot_value) != 0:
            raise TypeError(f"{type(self).__name__} expects a single threshold as pivot_value")
        return pivot_value

    def _check_models(self):
        for model in (self.left_model, self.right_model):
            if not isinstance(model, self.model_type):
                raise TypeError(f"{type(self).__name__} expects {self.model_type.__name__} side models, "
                                f"got {type(model).__name__}")

    def goes_left(self, value) -> bool:
        """Returns whether a sample with this pivot feature value goes to the left child.

        Args:
            value: Value of the sample at feature pivot_idx.

        Returns:
            True if the sample is routed to the left child.
        """
        return bool(self.goes_left_mask(value))

    def goes_left_mask(self, values) -> np.ndarray:
        """Vectorized goes_left: returns a boolean mask of the values routed to the left child.

        Args:
            values: Array of sample values at feature pivot_idx.

        Returns:
            Boolean array, True where the sample goes to the left child.
        """
        if self.categorical:
            return np.isin(values, self.pivot_value)
        return np.asarray(values) <= self.pivot_value

    def get_children(self) -> List[BaseNode]:
        """Returns this node's left and right children.

        Returns:
            List of [left_child, right_child].
        """
        return [self.left_child, self.right_child]

    def set_children(self, children: List[BaseNode]):
        """Replaces this node's left and right children.

        Args:
            children: List of [new_left_child, new_right_child].
        """
        self.left_child = children[0]
        self.right_child = children[1]

    def get_label(self) -> str:
        """Returns the full display label for this node.

        Returns:
            Label of the form "<NAME>\\nX<pivot_idx> > <pivot_value>", or
            "<NAME> X<pivot_idx>\\nidx ∉ <categories>" for a categorical split.
        """
        name = type(self).__name__.removesuffix("Node").upper()
        if self.categorical:
            return f"{name} X{self.pivot_idx}\nidx ∉ {self.pivot_value}"
        return f"{name}\nX{self.pivot_idx} > {self.pivot_value:.3g}"

    def to_dict(self) -> dict:
        """Serializes the node to a dictionary.

        Returns:
            Dictionary with base attributes plus pivot_idx, pivot_value,
            both children's class names and serialized forms, and both
            models' serialized forms.
        """
        node_dict = super().to_dict()
        node_dict.update({
            "pivot_idx": self.pivot_idx,
            "pivot_value": self.pivot_value,
            "left_child_class": self.left_child.__class__.__name__,
            "left_child_dict": self.left_child.to_dict(),
            "right_child_class": self.right_child.__class__.__name__,
            "right_child_dict": self.right_child.to_dict(),
            "left_model": self.left_model.to_dict(),
            "right_model": self.right_model.to_dict(),
        })
        return node_dict

    @classmethod
    def from_dict(cls, dic: dict) -> "SplitNode":
        """Reconstructs a SplitNode instance, including children and models.

        Args:
            dic: Dictionary previously produced by to_dict().

        Returns:
            A new instance of cls with children and side models restored.
        """
        left_child_class = cls.class_registry[dic["left_child_class"]]
        right_child_class = cls.class_registry[dic["right_child_class"]]

        obj = super().from_dict(dic)
        obj.pivot_idx = dic["pivot_idx"]
        obj.pivot_value = dic["pivot_value"]
        obj.left_child = left_child_class.from_dict(dic["left_child_dict"])
        obj.right_child = right_child_class.from_dict(dic["right_child_dict"])
        obj.left_model = node_model_from_dict(dic["left_model"])
        obj.right_model = node_model_from_dict(dic["right_model"])
        return obj


@BaseNode.register
class SplitCNode(SplitNode):
    """Categorical SplitNode: pivot_value lists the categories routed to the left child."""

    categorical = True


@BaseNode.register
class PlinNode(SplitNode):
    """PLIN: piecewise linear split with a SimpleLinearNodeModel on each side."""

    model_type = SimpleLinearNodeModel


@BaseNode.register
class BlinNode(SplitNode):
    """BLIN: broken linear split, like PLIN but the two models are continuous at pivot_value."""

    model_type = SimpleLinearNodeModel

    def _check_models(self):
        super()._check_models()
        left = self.left_model.predict(self.pivot_value)
        right = self.right_model.predict(self.pivot_value)
        if not np.isclose(left, right):
            raise ValueError(f"BlinNode models are not continuous at {self.pivot_value}: {left} vs {right}")


@BaseNode.register
class PconNode(SplitNode):
    """PCON: piecewise constant split with a ConstantNodeModel on each side."""

    model_type = ConstantNodeModel


@BaseNode.register
class PconcNode(PconNode):
    """PCONC: categorical PCON, pivot_value lists the categories routed to the left child."""

    categorical = True
