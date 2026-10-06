from imote.nodes.base_node import BaseNode
from typing import List

@BaseNode.register
class NoneNode(BaseNode):
    """A null object node representing an empty tree branch.
    """

    def __init__(self):
        super().__init__(indices=None, y_res=None, rss=0.0)

    def get_children(self) -> List[BaseNode]:
        """Returns this node's children.

        Returns:
            Empty list, a NoneNode has no children.
        """
        return []

    def get_label(self) -> str:
        """Returns the full display label for this node.

        Returns:
            The string "None".
        """
        return "None"

    def get_minimal_label(self) -> str:
        """Returns a short display label for this node.

        Returns:
            Empty string.
        """
        return ""

    def to_dict(self) -> dict:
        """Serializes the node to a dictionary.

        Returns:
            Dictionary with a single "node_type" marker.
        """
        return {"node_type": "NoneNode"}

    @classmethod
    def from_dict(cls, dic: dict) -> "NoneNode":
        """Reconstructs a NoneNode instance.

        Args:
            dic: Unused

        Returns:
            A new NoneNode instance.
        """
        obj = cls()
        obj.id = -1
        return obj