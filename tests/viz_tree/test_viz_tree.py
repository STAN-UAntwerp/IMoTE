import numpy as np
import pytest

from imote.nodes.collapsed_node import CollapsedNode
from imote.nodes.node_model import LinearNodeModel
from imote.nodes.none_node import NoneNode
from tests.helpers import X, leaf, make_pilot_tree, make_single_leaf_tree, make_split_tree, y
from imote.viz_tree.viz_tree import VizTree

TREES = [make_pilot_tree, make_split_tree]


def types(tree):
    return [type(n).__name__ for n in tree.nodes]


def collapsed_nodes(tree):
    return [n for n in tree.nodes if isinstance(n, CollapsedNode)]


# --- Construction and structure ---
@pytest.mark.parametrize("make_tree", TREES)
def test_ids_follow_node_order(make_tree):
    tree = make_tree()
    assert [n.id for n in tree.nodes] == list(range(len(tree.nodes)))


@pytest.mark.parametrize("make_tree", TREES)
def test_edges_connect_parent_to_child(make_tree):
    tree = make_tree()
    assert len(tree.edges) == len(tree.nodes) - 1
    assert all(child in parent.get_children() for parent, child in tree.edges)


@pytest.mark.parametrize("make_tree, depth, n_leafs", [
    (make_pilot_tree, 3, 3),
    (make_split_tree, 2, 3),
    (make_single_leaf_tree, 0, 1),
])
def test_depth_and_n_leafs(make_tree, depth, n_leafs):
    tree = make_tree()
    assert tree.get_depth() == depth
    assert tree.get_n_leafs() == n_leafs


# --- Prediction ---
def test_predict_pilot_tree(pilot_tree):
    x2 = X[:, 2]
    expected = np.where(np.isin(np.arange(8), [0, 2]), 3 * x2 + 1,
                        np.where(np.isin(np.arange(8), [1, 3]), -x2 + 3, x2 - 1))
    np.testing.assert_allclose(pilot_tree.predict(X), expected)


def test_predict_categorical_split(split_tree):
    # row 0 (x1=0) -> left leaf x0 + 1, row 1 (x1=1) -> right leaf 2
    np.testing.assert_allclose(split_tree.predict(X[:2]), [1.0, 2.0])


def test_predict_new_row_on_threshold(pilot_tree):
    # both thresholds use <=, so x0=3.5 and x2=0.5 go left/left -> 3*x2 + 1
    np.testing.assert_allclose(pilot_tree.predict(np.array([[3.5, 0, 0.5]])), [2.5])


def test_predict_through_collapsed_node_raises(pilot_tree):
    pilot_tree.collapse(pilot_tree.nodes[1])
    with pytest.raises(ValueError):
        pilot_tree.predict(X)


def test_y_hat_defaults_to_predict(pilot_tree):
    np.testing.assert_allclose(pilot_tree.y_hat, [1, 2, 1.6, 2.1, -0.9, -0.2, -0.7, -0.3])


def test_y_hat_given_is_stored():
    y_hat = np.arange(8, dtype=float)
    tree = VizTree(leaf(range(8), [0, 0, 1], 0.0), X, y, y_hat)
    assert tree.y_hat is y_hat


# --- Collapse / expand ---
def test_collapse(pilot_tree):
    blin = pilot_tree.nodes[2]
    assert pilot_tree.collapse(blin) == 2
    assert [type(c) for c in blin.get_children()] == [CollapsedNode, NoneNode]
    assert types(pilot_tree) == ["LinearNode", "PconNode", "BlinNode", "LeafNode", "CollapsedNode"]
    assert pilot_tree.get_depth() == 3


def test_expand_restores_nodes_and_ids(pilot_tree):
    original_types = types(pilot_tree)
    pilot_tree.collapse(pilot_tree.nodes[2])
    pilot_tree.expand(collapsed_nodes(pilot_tree)[0])
    assert types(pilot_tree) == original_types
    assert [n.id for n in pilot_tree.nodes] == list(range(6))


def test_collapse_over_collapsed_node_does_not_nest(pilot_tree):
    pilot_tree.collapse(pilot_tree.nodes[2])
    assert pilot_tree.collapse(pilot_tree.nodes[1]) == 4
    collapsed = collapsed_nodes(pilot_tree)
    assert len(collapsed) == 1
    assert not any(isinstance(n, CollapsedNode) for n in collapsed[0].parent.get_all_children())


@pytest.mark.parametrize("make_tree, collapse_id", [
    (make_pilot_tree, 2),
    (make_pilot_tree, 1),
    (make_split_tree, 1),
])
def test_expand_all_nodes(make_tree, collapse_id):
    tree = make_tree()
    original_types = types(tree)
    tree.collapse(tree.nodes[collapse_id])
    tree.expand_all_nodes()
    assert types(tree) == original_types
    assert [n.id for n in tree.nodes] == list(range(len(tree.nodes)))


# --- Prune ---
def test_prune_replaces_node_with_leaf(pilot_tree):
    pilot_tree.prune(pilot_tree.nodes[2])
    assert types(pilot_tree) == ["LinearNode", "PconNode", "LeafNode", "LeafNode"]
    new_leaf = pilot_tree.nodes[1].left_child
    assert new_leaf.id == 2
    # LinearNode(+1*x2) + Pcon.left_model(+1)
    assert isinstance(new_leaf.node_model, LinearNodeModel)
    np.testing.assert_array_equal(new_leaf.node_model.coefficients, [0, 0, 1])
    assert new_leaf.node_model.intercept == 1


def test_prune_recomputes_y_hat_and_contributions(pilot_tree):
    old_split = pilot_tree.split_contributions.copy()
    pilot_tree.prune(pilot_tree.nodes[2])
    np.testing.assert_allclose(pilot_tree.y_hat, [1, 2, 1.2, 1.9, -0.9, -0.2, -0.7, -0.3])
    assert pilot_tree.split_contributions.shape == X.shape
    assert not np.allclose(pilot_tree.split_contributions[:4], old_split[:4])


# --- Serialization ---
def assert_same_tree(restored, tree):
    assert restored.tree_id == tree.tree_id
    for attr in ["X_train", "y_train", "y_hat", "split_contributions", "linear_contributions"]:
        np.testing.assert_allclose(getattr(restored, attr), getattr(tree, attr))
    assert types(restored) == types(tree)
    assert [n.id for n in restored.nodes] == [n.id for n in tree.nodes]


@pytest.mark.parametrize("make_tree", TREES)
def test_round_trip(make_tree):
    tree = make_tree()
    restored = VizTree.from_dict(tree.to_dict())
    assert_same_tree(restored, tree)


def test_round_trip_with_collapsed_node(pilot_tree):
    original_types = types(pilot_tree)
    pilot_tree.collapse(pilot_tree.nodes[2])
    restored = VizTree.from_dict(pilot_tree.to_dict())
    assert_same_tree(restored, pilot_tree)
    restored.expand_all_nodes()
    assert types(restored) == original_types
