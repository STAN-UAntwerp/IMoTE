from typing import List
import numpy as np
from imote.nodes.base_node import BaseNode
from imote.nodes.node_model import node_model_from_dict, LinearNodeModel

@BaseNode.register
class LeafNode(BaseNode):
    """A terminal tree node holding the model used for its predictions.

    Attributes:
        node_model: The prediction model for samples reaching this leaf.
    """

    node_model: LinearNodeModel = None

    def __init__(self, indices: np.ndarray, y_res: np.ndarray, rss: float,
                 node_model: LinearNodeModel):
        """
        Args:
            indices: Row indices of the samples belonging to this node.
            y_res: Residuals of the target variable at this node.
            rss: Residual sum of squares at this node.
            node_model: The prediction model for samples reaching this leaf.
        """
        super().__init__(indices, y_res, rss)
        self.node_model = node_model

    def get_children(self) -> List[BaseNode]:
        """Returns this node's children.

        Returns:
            Empty list, a leaf node has no children.
        """
        return []

    def get_label(self) -> str:
        """Returns the full display label for this node.

        Returns:
            Label of the form "Leaf\\n<node_model label>".
        """
        return "Leaf\n" + self.node_model.get_label()

    def get_minimal_label(self) -> str:
        """Returns a short display label for this node.

        Returns:
            The string "L".
        """
        return "L"

    def to_dict(self) -> dict:
        """Serializes the node to a dictionary.

        Returns:
            Dictionary with base attributes plus the serialized node_model.
        """
        node_dict = super().to_dict()
        node_dict["node_model"] = self.node_model.to_dict()
        return node_dict

    @classmethod
    def from_dict(cls, dic: dict) -> "LeafNode":
        """Reconstructs a LeafNode instance, including its model.

        Args:
            dic: Dictionary previously produced by to_dict().

        Returns:
            A new LeafNode instance with its node_model restored.
        """
        obj = super().from_dict(dic)
        obj.node_model = node_model_from_dict(dic["node_model"])
        return obj