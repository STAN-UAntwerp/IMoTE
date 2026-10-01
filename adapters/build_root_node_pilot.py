from nodes.base_node import BaseNode
from nodes.leaf_node import LeafNode
from nodes.internal_node import LinearNode
from nodes.split_node import PconNode, BlinNode, PlinNode, PconcNode
from nodes.node_model import LinearNodeModel, ConstantNodeModel, SimpleLinearNodeModel
import numpy as np

def build_root_node_from_pilot(pilot_node, X_train, current_indices, current_y_res,
                               accumulated_coefficients, accumulated_intercept) -> BaseNode:

    if pilot_node.node == 'con' or pilot_node.node == 'END':
        return LeafNode(
            indices=current_indices,
            y_res=current_y_res,
            rss=pilot_node.Rt,
            node_model=LinearNodeModel(accumulated_coefficients, accumulated_intercept),
        )

    elif pilot_node.node == 'lin':
        pivot_idx = pilot_node.pivot[0]
        
        new_coefficients = accumulated_coefficients.copy()
        new_coefficients[pivot_idx] += pilot_node.lm_l[0]
        new_intercept = accumulated_intercept + pilot_node.lm_l[1]

        new_y_res = current_y_res - (pilot_node.lm_l[1] +
                                     pilot_node.lm_l[0] * X_train[current_indices, pivot_idx])

        child = build_root_node_from_pilot(pilot_node.left, X_train, current_indices, new_y_res,
                                           new_coefficients, new_intercept)

        return LinearNode(
            indices=current_indices,
            y_res=current_y_res,
            rss=pilot_node.Rt,
            pivot_idx=pivot_idx,
            linear_model=SimpleLinearNodeModel(pivot_idx, pilot_node.lm_l[0], pilot_node.lm_l[1]),
            child=child,
        )

    else:  # pcon, plin, blin, pconc
        pivot_idx, pivot_value = pilot_node.pivot
        if pilot_node.node == 'pconc':
            pivot_value = pilot_node.pivot_c
            left_mask = np.isin(X_train[current_indices, pivot_idx], pivot_value)
            right_mask = ~left_mask
        else:
            left_mask = X_train[current_indices, pivot_idx] <= pivot_value
            right_mask = ~left_mask
        left_indices, right_indices = current_indices.copy(), current_indices.copy()
        left_indices[current_indices] = left_mask
        right_indices[current_indices] = right_mask

        left_y_res = current_y_res[left_mask] - (pilot_node.lm_l[1] + pilot_node.lm_l[0] * X_train[left_indices, pivot_idx])
        left_coefficients = accumulated_coefficients.copy()
        left_coefficients[pivot_idx] += pilot_node.lm_l[0]
        left_intercept = accumulated_intercept + pilot_node.lm_l[1]

        right_y_res = current_y_res[right_mask] - (pilot_node.lm_r[1] + pilot_node.lm_r[0] * X_train[right_indices, pivot_idx])
        right_coefficients = accumulated_coefficients.copy()
        right_coefficients[pivot_idx] += pilot_node.lm_r[0]
        right_intercept = accumulated_intercept + pilot_node.lm_r[1]

        # Recurse to both children
        left_child = build_root_node_from_pilot(pilot_node.left, X_train, left_indices, left_y_res,
                                                left_coefficients, left_intercept)
        right_child = build_root_node_from_pilot(pilot_node.right, X_train, right_indices, right_y_res,
                                                 right_coefficients, right_intercept)

        if pilot_node.node == 'pcon':
            return PconNode(
                indices=current_indices,
                y_res=current_y_res,
                rss=pilot_node.Rt,
                pivot_idx=pivot_idx,
                pivot_value=pivot_value,
                left_child=left_child,
                right_child=right_child,
                left_model = ConstantNodeModel(pilot_node.lm_l[1]),
                right_model = ConstantNodeModel(pilot_node.lm_r[1]),
            )
        elif pilot_node.node == 'blin':
            return BlinNode(
                indices=current_indices,
                y_res=current_y_res,
                rss=pilot_node.Rt,
                pivot_idx=pivot_idx,
                pivot_value=pivot_value,
                left_child=left_child,
                right_child=right_child,
                left_model = SimpleLinearNodeModel(pivot_idx, pilot_node.lm_l[0], pilot_node.lm_l[1]),
                right_model = SimpleLinearNodeModel(pivot_idx, pilot_node.lm_r[0], pilot_node.lm_r[1]),
            )
        elif pilot_node.node == 'plin':
            return PlinNode(
                indices=current_indices,
                y_res=current_y_res,
                rss=pilot_node.Rt,
                pivot_idx=pivot_idx,
                pivot_value=pivot_value,
                left_child=left_child,
                right_child=right_child,
                left_model = SimpleLinearNodeModel(pivot_idx, pilot_node.lm_l[0], pilot_node.lm_l[1]),
                right_model = SimpleLinearNodeModel(pivot_idx, pilot_node.lm_r[0], pilot_node.lm_r[1]),
            )
        else:
            if pilot_node.node != 'pconc':
                raise ValueError("node type not recognized")
            return PconcNode(
                indices=current_indices,
                y_res=current_y_res,
                rss=pilot_node.Rt,
                pivot_idx=pivot_idx,
                pivot_value=pivot_value,
                left_child=left_child,
                right_child=right_child,
                left_model = ConstantNodeModel(pilot_node.lm_l[1]),
                right_model = ConstantNodeModel(pilot_node.lm_r[1]),
            )
