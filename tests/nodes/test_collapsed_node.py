import numpy as np
import pytest

from nodes.collapsed_node import CollapsedNode
from nodes.split_node import SplitNode


def test_init(tree):
    node = CollapsedNode(tree)
    assert node.n_nodes == 3
    assert node.parent_id == 0
    assert node.id == 2
    assert node.get_children() == []
    np.testing.assert_array_equal(node.indices, tree.indices)


def test_parent_is_a_copy(tree):
    node = CollapsedNode(tree)
    assert node.parent is not tree
    tree.set_children([tree.right_child, tree.left_child])
    assert node.parent.left_child.id == 2


def test_collapse_leaf_raises(leaf):
    with pytest.raises(ValueError):
        CollapsedNode(leaf)


def test_labels_are_strings(tree):
    node = CollapsedNode(tree)
    assert isinstance(node.get_label(), str) and node.get_label()
    assert isinstance(node.get_minimal_label(), str) and node.get_minimal_label()


def test_round_trip(tree):
    restored = CollapsedNode.from_dict(CollapsedNode(tree).to_dict())
    assert isinstance(restored, CollapsedNode)
    assert restored.id == 2
    assert restored.parent_id == 0
    assert restored.n_nodes == 3
    assert isinstance(restored.parent, SplitNode)
    assert [n.id for n in restored.parent.get_all_children()] == [2, 4, 3]
