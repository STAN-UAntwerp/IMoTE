import numpy as np
import pytest

from nodes.internal_node import LinearNode
from nodes.leaf_node import LeafNode
from nodes.node_model import SimpleLinearNodeModel
from tests.nodes.conftest import make_leaf


@pytest.fixture
def lin(leaf):
    node = LinearNode(np.array([0, 1]), np.array([0.2, -0.2]), 0.3, 3,
                      SimpleLinearNodeModel(3, 1.5, 0.25), leaf)
    node.set_id(0)
    return node


def test_children(lin, leaf):
    assert lin.get_children() == [leaf]
    new_leaf = make_leaf(5)
    lin.set_children([new_leaf])
    assert lin.get_children() == [new_leaf]


def test_labels_are_strings(lin):
    assert isinstance(lin.get_label(), str) and lin.get_label()
    assert isinstance(lin.get_minimal_label(), str) and lin.get_minimal_label()


def test_round_trip(lin):
    restored = LinearNode.from_dict(lin.to_dict())
    assert isinstance(restored, LinearNode)
    assert restored.id == 0
    assert restored.pivot_idx == 3
    assert isinstance(restored.linear_model, SimpleLinearNodeModel)
    assert (restored.linear_model.idx, restored.linear_model.coefficient, restored.linear_model.intercept) == (3, 1.5, 0.25)
    assert isinstance(restored.child, LeafNode)
    assert restored.child.id == 1
    assert restored.child is not lin.child
