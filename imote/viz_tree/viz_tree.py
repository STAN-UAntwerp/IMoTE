import numpy as np
from typing import List, Tuple, Type
from datetime import datetime

from imote.nodes.base_node import BaseNode
from imote.nodes.leaf_node import LeafNode
from imote.nodes.internal_node import InternalNode, LinearNode
from imote.nodes.collapsed_node import CollapsedNode
from imote.nodes.none_node import NoneNode
from imote.nodes.node_model import LinearNodeModel, ConstantNodeModel
from imote.nodes.split_node import PconNode, PconcNode, SplitNode, BlinNode, PlinNode
from imote.adapters.base_adapter import BaseAdapter

class VizTree:
    """VizTree is used to store the tree and visualise it in the application.
    It wraps a linked tree of BaseNode instances and provides the operations
    the visualization app needs: node/edge collection, prediction, collapsing/expanding
    subtrees, pruning, and serialization. It also saves the training data.

    Attributes:
        root_node: Root node of the linked tree.
        nodes: Flat list of all nodes in the tree.
        edges: List of (parent, child) node pairs for every edge in the tree.
        X_train: Training feature matrix used to fit the tree.
        y_train: Training target values used to fit the tree.
        y_hat: Predictions of the tree on X_train.
        tree_id: Identifier for this tree, defaults to a unique ID.
    """

    root_node: BaseNode
    nodes: List[BaseNode]
    edges: List[Tuple[BaseNode, BaseNode]]
    X_train: np.ndarray
    y_train: np.ndarray
    y_hat: np.ndarray
    tree_id: str

    def __init__(self,
                 root: BaseNode,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 y_hat: np.ndarray,
                 tree_id: str = None
                 ):
        """
        Args:
            root: Root node of the linked tree.
            X_train: Training feature matrix used to fit the tree.
            y_train: Training target values used to fit the tree.
            y_hat: Precomputed predictions of the tree on X_train, or
                None to compute them via predict().
            tree_id: Identifier for this tree. Defaults to the current
                timestamp (format "%d-%m-%y_%H-%M-%S") if not given.
        """
        if tree_id is None:
            self.tree_id = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        else:
            self.tree_id = tree_id

        self.X_train = X_train
        self.y_train = y_train

        self.root_node = root
        self.nodes = self.collect_nodes()
        self.edges = self.collect_edges()
        for i, node in enumerate(self.nodes):
            node.set_id(i)

        if y_hat is None:
            self.y_hat = self._calculate_y_hat()
        else:
            self.y_hat = y_hat
        self.split_contributions, self.linear_contributions = self._calculate_contributions()

    @classmethod
    def from_model(cls, adapter: Type[BaseAdapter], X_train: np.ndarray, y_train: np.ndarray, model) -> "VizTree":
        """Builds a VizTree from a fitted model via an adapter.

        Args:
            adapter: Object providing build_root_node(X_train, y_train, model)
                used to translate a model specific tree into a linked BaseNode tree.
            X_train: Training feature matrix used to fit the model.
            y_train: Training target values used to fit the model.
            model: The fitted model to convert to a VizTree.

        Returns:
            A new VizTree instance wrapping the model's tree.
        """
        root_node = adapter.build_root_node(X_train, y_train, model)
        y_hat = adapter.predict(X_train, model)
        return cls(root_node, X_train, y_train, y_hat)

    def collect_nodes(self) -> List[BaseNode]:
        """Collects all nodes in the tree.

        Returns:
            List of all nodes, with the root node first followed by
            all its descendants (depth-first order).
        """
        nodes = [self.root_node] + self.root_node.get_all_children()
        return [node for node in nodes if not isinstance(node, NoneNode)]

    def collect_edges(self) -> List[Tuple[BaseNode, BaseNode]]:
        """Collects all edges in the tree from self.nodes.

        Returns:
            List of (parent, child) tuples.
        """
        edges = []
        for node in self.nodes:
            for child in node.get_children():
                if isinstance(child, NoneNode):
                    continue
                edges.append((node, child))
        return edges

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts target values for each row of X by traversing the tree.

        Args:
            X: Feature matrix to predict on, one row per sample.

        Returns:
            Array of predicted values, one per row of X.

        Raises:
            ValueError: If traversal reaches node type that isn't recognized.
        """
        return self._predict_rows(self.root_node, X, np.arange(X.shape[0]))

    def _predict_rows(self, node: BaseNode, X: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Returns predictions for the given rows of X, routed down from node, recursively."""
        if len(rows) == 0:
            return np.empty(0)
        if isinstance(node, LeafNode):
            return node.node_model.predict(X[rows])
        if isinstance(node, LinearNode):
            return self._predict_rows(node.child, X, rows)
        if isinstance(node, SplitNode):
            go_left = node.goes_left_mask(X[rows, node.pivot_idx])
            preds = np.empty(len(rows))
            preds[go_left] = self._predict_rows(node.left_child, X, rows[go_left])
            preds[~go_left] = self._predict_rows(node.right_child, X, rows[~go_left])
            return preds
        raise ValueError(f"Can't predict node of type {type(node)}")


    def _calculate_y_hat(self) -> np.ndarray:
        """Computes predictions of the tree on X_train.

        Returns:
            Array of predicted values for X_train.
        """
        return self.predict(self.X_train)

    def _calculate_split_and_lin_contributions(self):
        X = self.X_train
        split_contributions = np.zeros_like(X, dtype=float)
        lin_contributions = np.zeros_like(X, dtype=float)
        for node in self.nodes:
            if isinstance(node, CollapsedNode):
                # raise ValueError("CollapsedNode cannot be used when calculating contributions")
                continue
            elif isinstance(node, InternalNode):
                pivot = node.pivot_idx
                if isinstance(node, LinearNode):
                    linear_predictions = node.linear_model.predict(X[node.child.indices, pivot])
                    lin_contributions[node.child.indices, pivot] += linear_predictions - np.mean(linear_predictions)
                    continue
                if isinstance(node, (PconcNode, PconNode)):
                    avg_pred_left = node.left_model.predict(np.array([0]))[0]
                    avg_pred_right = node.right_model.predict(np.array([0]))[0]
                elif isinstance(node, (BlinNode, PlinNode)):
                    linear_predictions_left = node.left_model.predict(X[node.left_child.indices, pivot])
                    linear_predictions_right = node.right_model.predict(X[node.right_child.indices, pivot])
                    avg_pred_left = np.mean(linear_predictions_left)
                    avg_pred_right = np.mean(linear_predictions_right)
                    lin_contributions[node.left_child.indices, pivot] += linear_predictions_left - avg_pred_left
                    lin_contributions[node.right_child.indices, pivot] += linear_predictions_right - avg_pred_right
                elif isinstance(node, SplitNode):
                    split_contributions, lin_contributions, _ = self._calculate_split_and_lin_contributions2_recursive(self.root_node)
                    return split_contributions, lin_contributions
                else:
                    raise NotImplementedError(
                        f"Node of class {node.__class__.__name__} is not implemented to calculate contributions")

                avg_pred = (avg_pred_left * np.sum(node.left_child.indices) + avg_pred_right * np.sum(
                    node.right_child.indices)) / np.sum(node.indices)
                split_contributions[node.left_child.indices, pivot] += avg_pred_left - avg_pred
                split_contributions[node.right_child.indices, pivot] += avg_pred_right - avg_pred
            elif isinstance(node, LeafNode):
                pass
            else:
                raise NotImplementedError(
                    f"Node of class {node.__class__.__name__} is not implemented to calculate contributions")

        return split_contributions, lin_contributions

    # For non-pilot trees
    def _calculate_split_and_lin_contributions2_recursive(self, node):
        if isinstance(node, LeafNode):
            node_model = node.node_model
            X_node = self.X_train[node.indices, :]
            avg_pred = np.mean(node_model.predict(X_node))

            split_contributions = np.zeros_like(self.X_train, dtype=float)
            lin_contributions = np.zeros_like(self.X_train, dtype=float)
            if isinstance(node_model, LinearNodeModel):
                lin_contributions[node.indices, :] += node_model.coefficients * (X_node - np.mean(X_node, axis=0))
            return split_contributions, lin_contributions, avg_pred
        elif isinstance(node, SplitNode):
            split_contributions_left, lin_contributions_left, avg_pred_left = (
                self._calculate_split_and_lin_contributions2_recursive(node.left_child))
            split_contributions_right, lin_contributions_right, avg_pred_right = (
                self._calculate_split_and_lin_contributions2_recursive(node.right_child))

            split_contributions = split_contributions_left + split_contributions_right
            lin_contributions = lin_contributions_left + lin_contributions_right

            avg_pred = (avg_pred_left * np.sum(node.left_child.indices) +
                        avg_pred_right * np.sum(node.right_child.indices)) / np.sum(node.indices)
            split_contributions[node.left_child.indices, node.pivot_idx] += avg_pred_left - avg_pred
            split_contributions[node.right_child.indices, node.pivot_idx] += avg_pred_right - avg_pred

            return split_contributions, lin_contributions, avg_pred
        else:
            raise NotImplementedError(
                f"Node of class {node.__class__.__name__} is not implemented to calculate contributions")

    def _calculate_contributions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes contributions of the tree on X_train.

        Returns:
            Tuple of split and linear contributions.
            Both matrices with the same shape as X_train.
        """
        return self._calculate_split_and_lin_contributions()

    def get_depth(self, node=None) -> int:
        """Computes the depth of the tree, or of a given subtree, recursively.

        Args:
            node: Root of the subtree to measure. Defaults to the
                tree's root_node if not given.

        Returns:
            Depth of the (sub)tree.
        """
        if node is None:
            node = self.root_node
        if isinstance(node, LeafNode):
            return 0
        elif isinstance(node, NoneNode):
            return -1
        elif isinstance(node, CollapsedNode):
            return self.get_depth(node.parent)-1
        return max([self.get_depth(node_i) for node_i in node.get_children()]) + 1

    def get_n_leafs(self) -> int:
        """Counts the number of leaf nodes in the tree.

        Returns:
            Number of LeafNode instances in nodes.
        """
        return sum(isinstance(node, LeafNode) for node in self.nodes)

    def collapse(self, parent_node: InternalNode) -> int:
        """Collapses a subtree in place, replacing it with a CollapsedNode.

        Args:
            parent_node: Root of the subtree to collapse.

        Returns:
            Number of nodes contained in the collapsed subtree.
        """
        for child in parent_node.get_all_children():
            if isinstance(child, CollapsedNode):
                self.expand(child)

        collapsed_node = CollapsedNode(parent_node)
        parent_node.set_children([collapsed_node, NoneNode()])

        self.nodes = self.collect_nodes()
        self.edges = self.collect_edges()

        return collapsed_node.n_nodes

    def expand(self, collapsed_node: CollapsedNode):
        """Expands a previously collapsed subtree back into the tree, in place.

        Args:
            collapsed_node: The CollapsedNode to expand.
        """
        for parent_node in self.nodes:
            if parent_node.id == collapsed_node.parent_id:
                break
        else:
            raise ValueError(f"Parent node {collapsed_node.parent_id} of the collapsed node is not in this tree")
        parent_node.set_children(collapsed_node.parent.get_children())

        self.nodes = self.collect_nodes()
        self.edges = self.collect_edges()

    def expand_all_nodes(self):
        """Expands every collapsed subtree in the tree, in place."""
        nodes_to_expand = []
        for node in self.nodes:
            if isinstance(node, CollapsedNode):
                nodes_to_expand.append(node)
        for node in nodes_to_expand:
            self.expand(node)

    def _get_path_to_node(self, to_find_node: BaseNode, current_node=None) -> List[BaseNode]:
        """Finds the path from the current_node to a target node, recursively.

        Args:
            to_find_node: The node to search for.
            current_node: Node to start the search from. Defaults to
                the tree's root_node if not given.

        Returns:
            List of nodes from current_node to to_find_node inclusive.
        """
        if current_node is None:
            current_node = self.root_node

        if current_node is to_find_node:
            return [current_node]

        if isinstance(current_node, LeafNode):
            return []

        for child in current_node.get_children():
            path = self._get_path_to_node(to_find_node, child)
            if path:
                return [current_node] + path
        return []

    def prune(self, prune_node: InternalNode):
        """Prunes a subtree in place, replacing it with a new leaf.

        Args:
            prune_node: Root of the subtree to prune.

        Raises:
            NotImplementedError: If a node on the path to prune_node,
                or its immediate parent, is not a LinearNode or SplitNode.
        """
        self.expand_all_nodes()
        node_path = self._get_path_to_node(prune_node)
        if len(node_path) < 2:
            raise ValueError("Can only prune a non-root node that is part of this tree")
        model = LinearNodeModel(coefficients = np.zeros(self.X_train.shape[1]), intercept = 0)

        for i in range(len(node_path)-1):
            node = node_path[i]
            if isinstance(node, LinearNode):
                model.add_model(node.linear_model)
            elif isinstance(node, SplitNode):
                next_node = node_path[i+1]
                if next_node is node.left_child:
                    model.add_model(node.left_model)
                else:
                    model.add_model(node.right_model)
            else:
                raise NotImplementedError

        if not np.any(model.coefficients):
            model = ConstantNodeModel(model.intercept)

        new_leaf_node = LeafNode(
            indices=prune_node.indices,
            y_res=prune_node.y_res,
            rss=prune_node.rss,
            node_model=model,
        )
        new_leaf_node.set_id(prune_node.id)

        parent = node_path[-2]
        parent.set_children([new_leaf_node if child is prune_node else child for child in parent.get_children()])

        self.nodes = self.collect_nodes()
        self.edges = self.collect_edges()
        self.y_hat = self._calculate_y_hat()
        self.split_contributions, self.linear_contributions = self._calculate_contributions()

    def to_dict(self) -> dict:
        """Serializes the tree to a dictionary.

        Returns:
            Dictionary with tree_id, X_train, y_train, y_hat,
            and the root node's class name and serialized form
            (which recursively serializes the whole linked tree).
        """
        return {
            "tree_id": self.tree_id,
            "X_train": self.X_train.tolist(),
            "y_train": self.y_train.tolist(),
            "y_hat": self.y_hat.tolist(),
            "split_contributions": self.split_contributions.tolist(),
            "linear_contributions": self.linear_contributions.tolist(),

            "root_node_class": self.root_node.__class__.__name__,
            "root_node_dict": self.root_node.to_dict(),
        }

    @classmethod
    def from_dict(cls, viz_dict:dict) -> "VizTree":
        """Reconstructs a VizTree instance from a dictionary.

        Args:
            viz_dict: Dictionary previously produced by to_dict().

        Returns:
            A new VizTree instance with its root node (and linked tree),
            and training data restored.
        """
        root_child_class = BaseNode.class_registry[viz_dict["root_node_class"]]

        obj = cls.__new__(cls)
        obj.tree_id = viz_dict["tree_id"]
        obj.X_train = np.array(viz_dict["X_train"])
        obj.y_train = np.array(viz_dict["y_train"])
        obj.y_hat = np.array(viz_dict["y_hat"])
        obj.split_contributions = np.array(viz_dict["split_contributions"])
        obj.linear_contributions = np.array(viz_dict["linear_contributions"])

        obj.root_node = root_child_class.from_dict(viz_dict["root_node_dict"])
        obj.nodes = obj.collect_nodes()
        obj.edges = obj.collect_edges()

        return obj