from types import SimpleNamespace

import numpy as np
import pytest

from adapters.pilot_adapter import PilotAdapter
from nodes.internal_node import LinearNode
from nodes.leaf_node import LeafNode
from nodes.split_node import BlinNode, PconcNode, PconNode, PlinNode
from tests.helpers import X, y
from viz_tree.viz_tree import VizTree


def con():
    return SimpleNamespace(node="con", Rt=0.1)


def lin(idx, coef, intercept, child):
    return SimpleNamespace(node="lin", pivot=(idx, None), lm_l=(coef, intercept), left=child, Rt=0.5)


def split(node, idx, value, lm_l, lm_r, pivot_c=None):
    return SimpleNamespace(node=node, pivot=(idx, value), pivot_c=pivot_c, lm_l=lm_l, lm_r=lm_r,
                           left=con(), right=con(), Rt=1.0)


def build(pilot_root):
    return PilotAdapter.build_root_node(X, y, SimpleNamespace(model_tree=pilot_root))


@pytest.mark.parametrize("pilot_root, cls", [
    (con(), LeafNode),
    (lin(2, 1.0, 0.0, con()), LinearNode),
    (split("pcon", 0, 3.5, (0, 1.0), (0, -1.0)), PconNode),
    (split("blin", 0, 3.5, (2.0, 0.0), (-2.0, 14.0)), BlinNode),
    (split("plin", 0, 3.5, (2.0, 0.0), (-2.0, 14.0)), PlinNode),
    (split("pconc", 1, None, (0, 1.0), (0, -1.0), pivot_c=[0]), PconcNode),
])
def test_node_type_mapping(pilot_root, cls):
    assert type(build(pilot_root)) is cls


@pytest.fixture
def lin_pcon_root():
    """lin(x2: 1*x2) -> pcon(x0 <= 3.5: left +1 / right -1) -> two leaves"""
    return build(lin(2, 1.0, 0.0, split("pcon", 0, 3.5, (0, 1.0), (0, -1.0))))


def test_leaf_model_accumulates_path(lin_pcon_root):
    pcon = lin_pcon_root.child
    np.testing.assert_array_equal(pcon.left_child.node_model.coefficients, [0, 0, 1])
    assert pcon.left_child.node_model.intercept == 1.0
    np.testing.assert_array_equal(pcon.right_child.node_model.coefficients, [0, 0, 1])
    assert pcon.right_child.node_model.intercept == -1.0
    tree = VizTree(lin_pcon_root, X, y, None)
    np.testing.assert_allclose(tree.predict(X), np.where(X[:, 0] <= 3.5, X[:, 2] + 1, X[:, 2] - 1))


def test_y_res_is_parent_residual_minus_parent_model(lin_pcon_root):
    pcon = lin_pcon_root.child
    np.testing.assert_allclose(pcon.y_res, y - X[:, 2])
    np.testing.assert_allclose(pcon.left_child.y_res, (y - X[:, 2])[:4] - 1)
    np.testing.assert_allclose(pcon.right_child.y_res, (y - X[:, 2])[4:] + 1)
