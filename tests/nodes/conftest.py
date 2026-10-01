import numpy as np
import pytest

from nodes.internal_node import LinearNode
from nodes.leaf_node import LeafNode
from nodes.node_model import ConstantNodeModel, LinearNodeModel, SimpleLinearNodeModel
from nodes.split_node import PconNode


def make_leaf(node_id, indices=(0, 1)):
    indices = np.array(indices)
    leaf = LeafNode(indices, np.zeros(len(indices)), 0.5, LinearNodeModel(np.array([0.0, 2.0]), 1.0))
    leaf.set_id(node_id)
    return leaf


@pytest.fixture
def leaf():
    return make_leaf(1)


@pytest.fixture
def tree():
    lin = LinearNode(np.array([0, 1]), np.array([0.1, -0.1]), 0.2, 1,
                     SimpleLinearNodeModel(1, 2.0, 0.5), make_leaf(3, (0, 1)))
    lin.set_id(2)
    root = PconNode(np.array([0, 1, 2, 3]), np.array([0.1, -0.1, 0.2, -0.2]), 1.0,
                    0, 1.5, lin, make_leaf(4, (2, 3)), ConstantNodeModel(1.0), ConstantNodeModel(2.0))
    root.set_id(0)
    return root
