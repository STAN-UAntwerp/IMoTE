from imote.nodes.base_node import BaseNode
from imote.nodes.leaf_node import LeafNode
from imote.nodes.internal_node import LinearNode
from imote.nodes.split_node import PconNode, BlinNode, PlinNode, PconcNode
from imote.nodes.node_model import LinearNodeModel, ConstantNodeModel, SimpleLinearNodeModel
import numpy as np
from .base_adapter import BaseAdapter, register_adapter


@register_adapter("Pilot")
class PilotAdapter(BaseAdapter):
    @staticmethod
    def build_root_node(X_train, y_train, model) -> BaseNode:
        tree = model.tree_summary()
        n_features = X_train.shape[1]
        root_indices = np.ones(X_train.shape[0], dtype=bool)
        if tree.parent_node_id[0]:
            raise ValueError('First node is not the root node')
        return PilotAdapter._build_recursive(
            tree, 0, X_train, root_indices, y_train,
            np.zeros(n_features), 0.0,
        )

    @staticmethod
    def _build_recursive(tree, node_index, X_train, current_indices, current_y_res,
                               accumulated_coefficients, accumulated_intercept) -> BaseNode:
        node_type = tree.node_type[node_index]
        if node_type == 'con' or node_type == 'END':
            leaf_intercept = accumulated_intercept + np.nan_to_num(tree.intercept_left[node_index])
            return LeafNode(
                indices=current_indices,
                y_res=current_y_res,
                rss=float(np.sum(current_y_res ** 2)),
                node_model=LinearNodeModel(accumulated_coefficients, leaf_intercept),
            )

        children_index = np.where(tree.parent_node_id == tree.node_id[node_index])[0]
        if node_type == 'lin':
            pivot_idx = int(tree.feature_index[node_index])
            coef = tree.slope_left[node_index]
            intercept = tree.intercept_left[node_index]

            new_coefficients = accumulated_coefficients.copy()
            new_coefficients[pivot_idx] += coef
            new_intercept = accumulated_intercept + intercept

            new_y_res = current_y_res - (intercept + coef * X_train[current_indices, pivot_idx])
            child_index = children_index[0]
            child = PilotAdapter._build_recursive(tree, child_index, X_train, current_indices, new_y_res,
                                                         new_coefficients, new_intercept)

            return LinearNode(
                indices=current_indices,
                y_res=current_y_res,
                rss=float(np.sum(current_y_res ** 2)),
                pivot_idx=pivot_idx,
                linear_model=SimpleLinearNodeModel(pivot_idx, coef, intercept),
                child=child,
            )
        elif node_type in ['pcon', 'pconc', 'blin', 'plin']:
            pivot_idx = int(tree.feature_index[node_index])
            coef_left = np.nan_to_num(tree.slope_left[node_index])
            intercept_left = np.nan_to_num(tree.intercept_left[node_index])
            coef_right = np.nan_to_num(tree.slope_right[node_index])
            intercept_right = np.nan_to_num(tree.intercept_right[node_index])

            if node_type == 'pconc':
                pivot_value = tree.pivot_values[node_index]
                left_mask = np.isin(X_train[current_indices, pivot_idx], pivot_value)
                right_mask = ~left_mask
            else:
                pivot_value = tree.split_value[node_index]
                left_mask = X_train[current_indices, pivot_idx] <= pivot_value
                right_mask = ~left_mask
            left_indices, right_indices = current_indices.copy(), current_indices.copy()
            left_indices[current_indices] = left_mask
            right_indices[current_indices] = right_mask

            left_y_res = current_y_res[left_mask] - (intercept_left + coef_left * X_train[left_indices, pivot_idx])
            left_coefficients = accumulated_coefficients.copy()
            left_coefficients[pivot_idx] += coef_left
            left_intercept = accumulated_intercept + intercept_left

            right_y_res = current_y_res[right_mask] - (intercept_right + coef_right * X_train[right_indices, pivot_idx])
            right_coefficients = accumulated_coefficients.copy()
            right_coefficients[pivot_idx] += coef_right
            right_intercept = accumulated_intercept + intercept_right

            # Recurse to both children
            left_child_index = children_index[0]
            right_child_index = children_index[1]
            left_child = PilotAdapter._build_recursive(tree, left_child_index, X_train, left_indices,
                                                              left_y_res, left_coefficients, left_intercept)
            right_child = PilotAdapter._build_recursive(tree, right_child_index, X_train, right_indices,
                                                               right_y_res, right_coefficients, right_intercept)

            if node_type == 'pcon':
                return PconNode(
                    indices=current_indices,
                    y_res=current_y_res,
                    rss=float(np.sum(current_y_res ** 2)),
                    pivot_idx=pivot_idx,
                    pivot_value=pivot_value,
                    left_child=left_child,
                    right_child=right_child,
                    left_model=ConstantNodeModel(intercept_left),
                    right_model=ConstantNodeModel(intercept_right),
                )
            elif node_type == 'pconc':
                return PconcNode(
                    indices=current_indices,
                    y_res=current_y_res,
                    rss=float(np.sum(current_y_res ** 2)),
                    pivot_idx=pivot_idx,
                    pivot_value=pivot_value,
                    left_child=left_child,
                    right_child=right_child,
                    left_model=ConstantNodeModel(intercept_left),
                    right_model=ConstantNodeModel(intercept_right),
                )
            elif node_type == 'blin':
                return BlinNode(
                    indices=current_indices,
                    y_res=current_y_res,
                    rss=float(np.sum(current_y_res ** 2)),
                    pivot_idx=pivot_idx,
                    pivot_value=pivot_value,
                    left_child=left_child,
                    right_child=right_child,
                    left_model=SimpleLinearNodeModel(pivot_idx, coef_left, intercept_left),
                    right_model=SimpleLinearNodeModel(pivot_idx, coef_right, intercept_right),
                )
            else:  # node_type == 'plin'
                return PlinNode(
                    indices=current_indices,
                    y_res=current_y_res,
                    rss=float(np.sum(current_y_res ** 2)),
                    pivot_idx=pivot_idx,
                    pivot_value=pivot_value,
                    left_child=left_child,
                    right_child=right_child,
                    left_model=SimpleLinearNodeModel(pivot_idx, coef_left, intercept_left),
                    right_model=SimpleLinearNodeModel(pivot_idx, coef_right, intercept_right),
                )
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    @staticmethod
    def load_model(model_path):
        raise NotImplementedError()

    @staticmethod
    def predict(X, model) -> np.ndarray:
        return model.predict(X)
