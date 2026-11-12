import numpy as np
from imote.adapters.base_adapter import BaseAdapter, register_adapter
from imote.adapters.bcpilot_adapter import _working_response
from imote.nodes.base_node import BaseNode
from imote.nodes.leaf_node import LeafNode
from imote.nodes.split_node import SplitNode
from imote.nodes.node_model import LinearNodeModel, NoneNodeModel


def _log_odds(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


class _DecisionTreeAdapter(BaseAdapter):
    """Shared base of the adapters for fitted sklearn decision trees.

    Every node is treated as if the tree stopped there: it predicts one constant value for its samples,
    and its y_res are the residuals of that value.
    """

    @staticmethod
    def _node_value(y: np.ndarray) -> float:
        """Constant prediction of a node with target values y."""
        raise NotImplementedError

    @staticmethod
    def _residuals(y: np.ndarray, value: float) -> np.ndarray:
        """y_res of a node with target values y that predicts value."""
        raise NotImplementedError

    @classmethod
    def build_root_node(cls, X_train, y_train, model) -> BaseNode:
        # The samples of a node come from sklearn, which splits on float32 values and can send missing values left.
        node_samples = model.decision_path(X_train).tocsc()
        return cls._build_recursive(model.tree_, 0, node_samples, y_train)

    @classmethod
    def _build_recursive(cls, tree, node_id, node_samples, y_train) -> BaseNode:
        current_indices = node_samples[:, node_id].toarray().ravel().astype(bool)
        current_y = y_train[current_indices]
        value = cls._node_value(current_y)
        y_res = cls._residuals(current_y, value)
        rss = float(np.sum(y_res ** 2))

        left_id, right_id = tree.children_left[node_id], tree.children_right[node_id]
        if left_id == -1:  # sklearn's TREE_LEAF sentinel
            node = LeafNode(
                indices=current_indices,
                y_res=y_res,
                rss=rss,
                node_model=LinearNodeModel(np.zeros(tree.n_features), value),
            )
        else:
            node = SplitNode(
                indices=current_indices,
                y_res=y_res,
                rss=rss,
                pivot_idx=int(tree.feature[node_id]),
                pivot_value=float(tree.threshold[node_id]),
                left_child=cls._build_recursive(tree, left_id, node_samples, y_train),
                right_child=cls._build_recursive(tree, right_id, node_samples, y_train),
                left_model=NoneNodeModel(),
                right_model=NoneNodeModel(),
            )
        if cls.task == "classification":
            node.log_odds = np.full(len(current_y), value)
        return node

    @staticmethod
    def load_model(model_path):
        raise NotImplementedError()


@register_adapter("DecisionTreeRegressor")
class DecisionTreeRegressorAdapter(_DecisionTreeAdapter):
    """Adapter for sklearn.tree.DecisionTreeRegressor, a node predicts the mean of its samples."""

    @staticmethod
    def _node_value(y):
        return float(np.mean(y))

    @staticmethod
    def _residuals(y, value):
        return y - value

    @staticmethod
    def predict(X, model) -> np.ndarray:
        return model.predict(X)


@register_adapter("DecisionTreeClassifier")
class DecisionTreeClassifierAdapter(_DecisionTreeAdapter):
    """Adapter for a binary sklearn.tree.DecisionTreeClassifier fitted on a 0/1 target.

    A node predicts the log-odds of the fraction of class 1 among its samples.
    """

    task = "classification"

    @staticmethod
    def _node_value(y):
        return float(_log_odds(np.mean(y)))

    @staticmethod
    def _residuals(y, value):
        return _working_response(y, np.full(len(y), value))

    @staticmethod
    def predict(X, model) -> np.ndarray:
        return _log_odds(model.predict_proba(X)[:, 1])
