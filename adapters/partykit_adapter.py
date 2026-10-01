import os
import json
import numpy as np
from adapters.base_adapter import BaseAdapter, register_adapter
from adapters.adapter_utils import split_by_categories, split_by_threshold
from nodes.base_node import BaseNode
from nodes.leaf_node import LeafNode
from nodes.split_node import SplitNode, SplitCNode
from nodes.node_model import LinearNodeModel, NoneNodeModel

@register_adapter('Partykit')
class PartyKitAdapter(BaseAdapter):
    """Adapter that builds a BaseNode linked tree from an R `partykit` package model export."""

    @staticmethod
    def load_model(model_path: str) -> dict:
        """Loads a partykit model export from a JSON file.

        Args:
            model_path: Path to the JSON file containing the exported
                partykit model.

        Returns:
            The parsed model as a dict.

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No such file path to load model: {model_path}")  # The standard exception for a missing file
        with open(model_path) as f:
            model = json.load(f)

        return model

    @staticmethod
    def build_root_node(X_train: np.ndarray, y_train: np.ndarray, model) -> BaseNode:
        """Builds the BaseNode linked tree representing a partykit model.

        Args:
            X_train: Training feature matrix the model was fit on.
            y_train: Training target values the model was fit on.
            model: Parsed partykit model, as returned by load_model().

        Returns:
            Root BaseNode of the reconstructed tree.
        """
        root_indices = np.ones(X_train.shape[0], dtype=bool)
        node_list = model['nodes']
        feature_names = model['names'][1:]
        return PartyKitAdapter._build_recursive(node_list, feature_names, node_list[0], X_train, y_train, root_indices)

    @staticmethod
    def _build_recursive(node_list, feature_names, node, X_train, y_train, current_indices) -> BaseNode:
        """Recursively builds a BaseNode (sub)tree from partykit node data.

        Args:
            node_list: Full flat list of partykit node dicts for the model.
            feature_names: Feature names, in the same column order as X_train.
            node: The partykit node dict to convert at this recursion step.
            X_train: Training feature matrix the model was fit on.
            y_train: Training target values the model was fit on.
            current_indices: Boolean mask over X_train's rows selecting
                the samples that reach this node.

        Returns:
            A BaseNode linked (sub)tree.

        Raises:
            ValueError: If node has neither a "breaks" nor an "index"
                key, so its split type can't be determined.
        """
        current_y_res = y_train[current_indices]
        n_features = X_train.shape[1]

        if node["is_terminal"]:
            coefficients = np.array([
                node["coefficients"].get(name, 0.0)
                for name in feature_names
            ], dtype=float)

            node_model = LinearNodeModel(coefficients, node["coefficients"]['(Intercept)'])

            return LeafNode(
                indices=current_indices,
                y_res=current_y_res,
                rss=-1, #TODO
                node_model=node_model,
            )
        else:
            pivot_idx = feature_names.index(node["split_var"])
            left_node = node_list[node["kids"][0]]
            right_node = node_list[node["kids"][1]]
            if "breaks" in node:
                pivot_value = node["breaks"]
                left_indices, right_indices = split_by_threshold(X_train, current_indices, pivot_idx, pivot_value)

                left_child = PartyKitAdapter._build_recursive(
                    node_list, feature_names, left_node, X_train, y_train, left_indices)
                right_child = PartyKitAdapter._build_recursive(
                    node_list, feature_names, right_node, X_train, y_train, right_indices)

                return SplitNode(
                    indices=current_indices,
                    y_res=current_y_res,
                    rss=-1, #TODO
                    pivot_idx=pivot_idx,
                    pivot_value=pivot_value,
                    left_child=left_child,
                    right_child=right_child,
                    left_model=NoneNodeModel(),
                    right_model=NoneNodeModel(),
                )
            elif "index" in node:
                left_categories = [lvl for lvl, idx in zip(node["levels"], node["index"]) if idx == 1]
                left_indices, right_indices = split_by_categories(X_train, current_indices, pivot_idx, left_categories)

                left_child = PartyKitAdapter._build_recursive(
                    node_list, feature_names, left_node, X_train, y_train, left_indices)
                right_child = PartyKitAdapter._build_recursive(
                    node_list, feature_names, right_node, X_train, y_train, right_indices)
                return SplitCNode(
                    indices=current_indices,
                    y_res=current_y_res,
                    rss=-1, #TODO
                    pivot_idx=pivot_idx,
                    pivot_value=left_categories,
                    left_child=left_child,
                    right_child=right_child,
                    left_model=NoneNodeModel(),
                    right_model=NoneNodeModel(),
                )
            else:
                raise ValueError(f"Unknown node: {node}")