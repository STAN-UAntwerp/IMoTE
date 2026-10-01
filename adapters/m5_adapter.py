import numpy as np
from nodes.base_node import BaseNode
from .build_root_node_m5 import build_root_node_from_m5
from .base_adapter import BaseAdapter, register_adapter

@register_adapter("M5")
class M5Adapter(BaseAdapter):
    @staticmethod
    def build_root_node(X_train, y_train, model) -> BaseNode:
        return build_root_node_from_m5(model, X_train, y_train)

    @staticmethod
    def load_model(model_path):
        raise NotImplementedError()

    @staticmethod
    def predict(X, m5_model) -> np.ndarray:
        return m5_model.predict(X)