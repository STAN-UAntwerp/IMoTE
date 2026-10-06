import numpy as np
import pytest

from imote.nodes.leaf_node import LeafNode
from imote.nodes.node_model import ConstantNodeModel, LinearNodeModel


def test_has_no_children(leaf):
    assert leaf.get_children() == []


def test_labels_are_strings(leaf):
    assert isinstance(leaf.get_label(), str) and leaf.get_label()
    assert isinstance(leaf.get_minimal_label(), str) and leaf.get_minimal_label()


@pytest.mark.parametrize("model", [ConstantNodeModel(4.0), LinearNodeModel(np.array([1.0, 3.0]), 0.5)])
def test_round_trip(model):
    node = LeafNode(np.array([2, 5]), np.array([0.1, 0.2]), 0.05, model)
    node.set_id(9)

    restored = LeafNode.from_dict(node.to_dict())
    assert isinstance(restored, LeafNode)
    assert restored.id == 9
    assert restored.rss == 0.05
    np.testing.assert_array_equal(restored.indices, node.indices)
    np.testing.assert_array_equal(restored.y_res, node.y_res)
    assert type(restored.node_model) is type(model)
    X = np.array([[1.0, 2.0]])
    np.testing.assert_array_equal(restored.node_model.predict(X), model.predict(X))
