import numpy as np
import pytest
from m5py import M5Prime

from imote.adapters.m5_adapter import M5Adapter
from imote.nodes.leaf_node import LeafNode
from imote.nodes.node_model import ConstantNodeModel, LinearNodeModel
from imote.nodes.split_node import SplitNode
from imote.viz_tree.viz_tree import VizTree


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(200, 3))
    y = np.where(X[:, 0] > 0.5, 3 * X[:, 1], -2 * X[:, 2]) + rng.normal(0, 0.05, 200)
    return X, y


# M5 has no numba and fits in ~0.02s, so real fit
def test_viz_tree_matches_model(data):
    X, y = data
    model = M5Prime(max_depth=4).fit(X, y)
    tree = VizTree.from_model(M5Adapter, X, y, model)
    for node in tree.nodes:
        if isinstance(node, LeafNode):
            assert isinstance(node.node_model, (LinearNodeModel, ConstantNodeModel))
        else:
            assert isinstance(node, SplitNode)
    np.testing.assert_allclose(tree.predict(X), model.predict(X), atol=1e-6)
