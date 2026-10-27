"""
Node type combining a chain of successive LinearNode (LIN) nodes into one node.

Used to display a run of single feature linear (LIN) nodes as a
single combined linear step.

Not used for initialising a tree structure.
"""

from typing import List

from imote.nodes.base_node import BaseNode
from imote.nodes.label_formatter import LabelFormatter
from imote.nodes.internal_node import LinearNode

class CombinedLinNode(BaseNode):
    child: BaseNode
    pivot_indices: List[int]
    lin_coefficients: dict[int, float]
    intercept: float
    true_child: BaseNode

    def __init__(self, nodes: List[LinearNode]):
        super().__init__(nodes[-1].indices, nodes[-1].y_res, nodes[-1].rss)
        pivot_indices = []
        lin_coefficients = {}
        intercept = 0
        for node in nodes:
            pivot_indices.append(node.pivot_idx)
            lin_coefficients[node.pivot_idx] = node.linear_model.coefficient
            intercept += node.linear_model.intercept

        self.child = nodes[-1].child
        #nodes[-1].child = None
        self.pivot_indices = pivot_indices
        self.lin_coefficients = lin_coefficients
        self.intercept = intercept
        self.id = nodes[0].id
    #     self.true_child = nodes[0]
    #
    # def expand(self):
    #     return self.true_child

    def get_children(self):
        return [self.child]

    def get_label(self, fmt: LabelFormatter) -> str:
        return "LIN\n" + ", ".join(fmt.feature(i) for i in self.pivot_indices)

    def get_minimal_label(self, fmt: LabelFormatter) -> str:
        return f"{fmt.feature(self.pivot_indices[0])}+"

    def get_description(self, fmt: LabelFormatter) -> str:
        return fmt.expression([(i, self.lin_coefficients[i]) for i in self.pivot_indices], self.intercept)

    # def to_dict(self):
    #     node_dict = super().to_dict()
    #     node_dict.update({
    #         "pivot_indices": self.pivot_indices,
    #         "lin_coefficients": self.lin_coefficients,
    #         "intercept": self.intercept,
    #         "child_class": self.child.__class__.__name__,
    #         "child_dict": self.child.to_dict(),
    #         # "true_child_class": self.true_child.__class__.__name__,
    #         # "true_child_dict": self.true_child.to_dict()
    #     })
    #     return node_dict

    # @classmethod
    # def from_dict(cls, dic: dict) -> "LinearNode":
    #     child_class = cls.class_registry[dic["child_class"]]
    #     true_child_class = cls.class_registry[dic["true_child_class"]]
    #
    #     obj = super().from_dict(dic)
    #     obj.pivot_indices = dic["pivot_indices"]
    #     obj.lin_coefficients = dic["lin_coefficients"]
    #     obj.intercept = dic["intercept"]
    #     obj.child = child_class.from_dict(dic["child_dict"])
    #     obj.true_child = true_child_class.from_dict(dic["true_child_dict"])
    #     return obj