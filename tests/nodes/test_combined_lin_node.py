import numpy as np
import pytest

from imote.nodes.combined_lin_node import CombinedLinNode
from imote.nodes.internal_node import LinearNode
from imote.nodes.node_model import SimpleLinearNodeModel


@pytest.fixture
def chain(leaf):
    last = LinearNode(np.array([0, 1]), np.array([0.3, -0.3]), 0.2, 4,
                      SimpleLinearNodeModel(4, -1.0, 0.25), leaf)
    last.set_id(6)
    first = LinearNode(np.array([0, 1, 2]), np.array([0.5, -0.5, 0.1]), 0.9, 2,
                       SimpleLinearNodeModel(2, 3.0, 0.5), last)
    first.set_id(5)
    return [first, last]


def test_init(chain, leaf):
    node = CombinedLinNode(chain)
    assert node.id == 5
    assert node.pivot_indices == [2, 4]
    assert node.lin_coefficients == {2: 3.0, 4: -1.0}
    assert node.intercept == 0.75
    np.testing.assert_array_equal(node.indices, chain[-1].indices)
    assert node.get_children() == [leaf]


def test_labels_are_strings(chain):
    node = CombinedLinNode(chain)
    assert isinstance(node.get_label(), str) and node.get_label()
    assert isinstance(node.get_minimal_label(), str) and node.get_minimal_label()
