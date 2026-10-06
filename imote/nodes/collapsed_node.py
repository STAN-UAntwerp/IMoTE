from imote.nodes.base_node import BaseNode

@BaseNode.register
class CollapsedNode(BaseNode):
    parent: BaseNode
    n_nodes: int

    def __init__(self, parent_node: BaseNode):
        super().__init__(parent_node.indices, parent_node.y_res, parent_node.rss)
        copy_parent = parent_node.__class__.from_dict(parent_node.to_dict())
        self.parent_id = parent_node.id
        self.parent = copy_parent
        all_children = self.parent.get_all_children()
        self.n_nodes = len(all_children)
        if self.n_nodes == 0:
            raise ValueError("Can't collapse node without children.")
        self.id = all_children[0].id

    def get_children(self):
        return []

    def get_label(self) -> str:
        return f"{self.n_nodes} collapsed\nnodes"

    def get_minimal_label(self) -> str:
        return f"+{self.n_nodes}"

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict.update({
            "parent_id": self.parent_id,
            "parent_class": self.parent.__class__.__name__,
            "parent_dict": self.parent.to_dict(),
            "n_nodes": self.n_nodes,
        })
        return node_dict

    @classmethod
    def from_dict(cls, dic):
        parent_class = cls.class_registry[dic["parent_class"]]

        obj = super().from_dict(dic)
        obj.parent_id = dic["parent_id"]
        obj.n_nodes = dic["n_nodes"]
        obj.parent = parent_class.from_dict(dic["parent_dict"])
        return obj
