import numpy as np
from nodes.base_node import BaseNode
from .build_root_node_pilot import build_root_node_from_pilot
from .base_adapter import BaseAdapter, register_adapter

@register_adapter("Pilot")
class PilotAdapter(BaseAdapter):
    @staticmethod
    def build_root_node(X_train, y_train, model) -> BaseNode:
        n_features = X_train.shape[1]
        root_indices = np.ones(X_train.shape[0], dtype=bool)
        return build_root_node_from_pilot(
            model.model_tree, X_train, root_indices, y_train,
            np.zeros(n_features), 0.0,
        )

    @staticmethod
    def load_model(model_path):
        raise NotImplementedError()

    @staticmethod
    def predict(X, pilot_model) -> np.ndarray:
        return pilot_model.predict(X)