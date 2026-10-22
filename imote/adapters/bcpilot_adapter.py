from imote.nodes.base_node import BaseNode
from imote.nodes.leaf_node import LeafNode
from imote.nodes.internal_node import LinearNode
from imote.nodes.split_node import PconNode, BlinNode, PlinNode, PconcNode
from imote.nodes.node_model import LinearNodeModel, ConstantNodeModel, SimpleLinearNodeModel
import numpy as np
from .base_adapter import BaseAdapter, register_adapter


def _sigmoid(f):
    return 1 / (1 + np.exp(-np.clip(f, -50, 50)))


def _working_response(y, f):
    """IRLS working response (y - p) / (p (1 - p)), the pseudo response a bcPilot node is fit on."""
    p = _sigmoid(f)
    h = p * (1 - p)
    z = np.zeros_like(f)
    nonzero = h != 0
    z[nonzero] = (y[nonzero] - p[nonzero]) / h[nonzero]
    return z


@register_adapter("BcPilot")
class BcPilotAdapter(BaseAdapter):
    """Adapter for the binary classification PILOT in imote.bcpilot.

    All node models are on the log-odds scale: a leaf predicts log(p / (1 - p)).
    The y_res of a node is the working response the node is fit on.
    """

    @staticmethod
    def build_root_node(X_train, y_train, model) -> BaseNode:
        if model.warm_start is not None:
            raise NotImplementedError("bcPilot trees fitted with warm_start are not supported yet")
        n_features = X_train.shape[1]
        root_indices = np.ones(X_train.shape[0], dtype=bool)
        f_root = BcPilotAdapter._baseline(model, X_train.shape[0])
        return BcPilotAdapter._build_recursive(
            model, model.model_tree, X_train, y_train, root_indices, f_root,
            np.zeros(n_features), float(f_root[0]) if len(f_root) else 0.0,
        )

    @staticmethod
    def _baseline(model, n_samples) -> np.ndarray:
        if getattr(model, "init", "zero") == "logodds":
            return np.full(n_samples, model.logodds)
        return np.zeros(n_samples)

    @staticmethod
    def _contribution(model, tree, x, go_left=True) -> np.ndarray:
        """Log-odds that a bcPilot node adds for feature values x."""
        if tree.node == "pconc":
            contribution = np.full(len(x), tree.intercept_left if go_left else tree.intercept_right, dtype=float)
        elif go_left:
            x_truncated = np.clip(x, tree.interval[0], tree.interval[1])
            contribution = tree.slope_left * x_truncated + tree.intercept_left
        else:
            x_truncated = np.minimum(x, tree.interval[1])
            contribution = tree.slope_right * x_truncated + tree.intercept_right
        return model._scale_and_clip(contribution)

    @staticmethod
    def _build_recursive(model, tree, X_train, y_train, current_indices, f,
                         accumulated_coefficients, accumulated_intercept) -> BaseNode:
        """Builds the node for bcPilot node `tree`.

        f holds the log-odds of the samples in current_indices before this node.
        """
        current_y = y_train[current_indices]
        y_res = _working_response(current_y, f)
        rss = float(np.sum(y_res ** 2))
        node_type = tree.node

        if node_type == "END":
            return LeafNode(
                indices=current_indices,
                y_res=y_res,
                rss=rss,
                node_model=LinearNodeModel(accumulated_coefficients, accumulated_intercept),
            )

        if node_type == "con":
            # a bcPilot con node can continue in the tree, fold its constant into the nodes below
            # Maybe need to make new con node type
            new_f = f + BcPilotAdapter._contribution(model, tree, np.zeros(len(f)))
            return BcPilotAdapter._build_recursive(
                model, tree.left, X_train, y_train, current_indices, new_f,
                accumulated_coefficients, accumulated_intercept + tree.intercept_left,
            )

        pivot_idx = int(tree.pivot[0])
        x = X_train[current_indices, pivot_idx]

        if node_type in ("lin", "tlin"):
            coef = tree.slope_left
            intercept = tree.intercept_left
            new_coefficients = accumulated_coefficients.copy()
            new_coefficients[pivot_idx] += coef
            new_f = f + BcPilotAdapter._contribution(model, tree, x)
            child = BcPilotAdapter._build_recursive(
                model, tree.left, X_train, y_train, current_indices, new_f,
                new_coefficients, accumulated_intercept + intercept,
            )
            return LinearNode(
                indices=current_indices,
                y_res=y_res,
                rss=rss,
                pivot_idx=pivot_idx,
                linear_model=SimpleLinearNodeModel(pivot_idx, coef, intercept),
                child=child,
            )

        if node_type not in ("pcon", "pconc", "blin", "plin"):
            raise ValueError(f"Unknown node type: {node_type}")

        coef_left, intercept_left = tree.slope_left, tree.intercept_left
        coef_right, intercept_right = tree.slope_right, tree.intercept_right
        if node_type == "pconc":
            pivot_value = list(tree.pivot_categorical)
            left_mask = np.isin(x, pivot_value)
        else:
            pivot_value = float(tree.pivot[1])
            left_mask = x <= pivot_value
        left_indices, right_indices = current_indices.copy(), current_indices.copy()
        left_indices[current_indices] = left_mask
        right_indices[current_indices] = ~left_mask

        left_f = f[left_mask] + BcPilotAdapter._contribution(model, tree, x[left_mask], go_left=True)
        right_f = f[~left_mask] + BcPilotAdapter._contribution(model, tree, x[~left_mask], go_left=False)

        left_coefficients = accumulated_coefficients.copy()
        left_coefficients[pivot_idx] += coef_left
        right_coefficients = accumulated_coefficients.copy()
        right_coefficients[pivot_idx] += coef_right

        left_child = BcPilotAdapter._build_recursive(model, tree.left, X_train, y_train, left_indices, left_f,
                                                     left_coefficients, accumulated_intercept + intercept_left)
        right_child = BcPilotAdapter._build_recursive(model, tree.right, X_train, y_train, right_indices, right_f,
                                                      right_coefficients, accumulated_intercept + intercept_right)

        if node_type in ("pcon", "pconc"):
            node_cls = PconNode if node_type == "pcon" else PconcNode
            left_model, right_model = ConstantNodeModel(intercept_left), ConstantNodeModel(intercept_right)
        else:
            node_cls = PlinNode if node_type == "plin" else BlinNode
            left_model = SimpleLinearNodeModel(pivot_idx, coef_left, intercept_left)
            right_model = SimpleLinearNodeModel(pivot_idx, coef_right, intercept_right)
        return node_cls(
            indices=current_indices,
            y_res=y_res,
            rss=rss,
            pivot_idx=pivot_idx,
            pivot_value=pivot_value,
            left_child=left_child,
            right_child=right_child,
            left_model=left_model,
            right_model=right_model,
        )

    @staticmethod
    def load_model(model_path):
        raise NotImplementedError()

    @staticmethod
    def predict(X, model) -> np.ndarray:
        proba = model.predict_proba(X)[:, 1] # Predict and select proba of class 1
        proba = np.clip(proba, 1e-15, 1 - 1e-15)
        return np.log(proba / (1 - proba))
