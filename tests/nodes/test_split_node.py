import numpy as np
import pytest

from nodes.internal_node import LinearNode
from nodes.leaf_node import LeafNode
from nodes.node_model import ConstantNodeModel, NoneNodeModel, SimpleLinearNodeModel
from nodes.none_node import NoneNode
from nodes.split_node import BlinNode, PconcNode, PconNode, PlinNode, SplitCNode, SplitNode
from tests.nodes.conftest import make_leaf

VALID = {
    SplitNode: (1.5, NoneNodeModel(), NoneNodeModel()),
    SplitCNode: ([0, 3], NoneNodeModel(), NoneNodeModel()),
    PlinNode: (1.5, SimpleLinearNodeModel(2, 1.0, 0.0), SimpleLinearNodeModel(2, -1.0, 4.0)),
    BlinNode: (1.5, SimpleLinearNodeModel(2, 1.0, 0.0), SimpleLinearNodeModel(2, -1.0, 3.0)),
    PconNode: (1.5, ConstantNodeModel(1.0), ConstantNodeModel(2.0)),
    PconcNode: ([0, 3], ConstantNodeModel(1.0), ConstantNodeModel(2.0)),
}
CLASSES = list(VALID)


def make_split(cls, pivot_value=None, left_model=None, right_model=None, left=None, right=None):
    default_pivot, default_left, default_right = VALID[cls]
    node = cls(np.array([0, 1, 2, 3]), np.zeros(4), 1.0, 2,
               default_pivot if pivot_value is None else pivot_value,
               left or make_leaf(1), right or make_leaf(2),
               left_model or default_left, right_model or default_right)
    node.set_id(0)
    return node


def test_children():
    left, right = make_leaf(1), make_leaf(2)
    node = make_split(SplitNode, left=left, right=right)
    assert node.get_children() == [left, right]

    new_left, new_right = make_leaf(3), make_leaf(4)
    node.set_children([new_left, new_right])
    assert node.get_children() == [new_left, new_right]


@pytest.mark.parametrize("cls", CLASSES)
def test_labels_are_strings(cls):
    node = make_split(cls)
    assert isinstance(node.get_label(), str) and node.get_label()
    assert isinstance(node.get_minimal_label(), str) and node.get_minimal_label()


@pytest.mark.parametrize("cls, wrong_model", [
    (SplitNode, ConstantNodeModel(1.0)),
    (SplitCNode, ConstantNodeModel(1.0)),
    (PlinNode, ConstantNodeModel(1.0)),
    (BlinNode, NoneNodeModel()),
    (PconNode, SimpleLinearNodeModel(2, 1.0, 0.0)),
    (PconcNode, NoneNodeModel()),
])
def test_wrong_model_type_raises(cls, wrong_model):
    with pytest.raises(TypeError):
        make_split(cls, left_model=wrong_model)


def test_blin_must_be_continuous():
    with pytest.raises(ValueError):
        make_split(BlinNode, right_model=SimpleLinearNodeModel(2, -1.0, 4.0))


@pytest.mark.parametrize("cls", [SplitNode, PlinNode, BlinNode, PconNode])
def test_numeric_split_rejects_list_pivot(cls):
    with pytest.raises(TypeError):
        make_split(cls, pivot_value=[0, 3])


@pytest.mark.parametrize("cls", [SplitCNode, PconcNode])
def test_categorical_split_rejects_threshold_pivot(cls):
    with pytest.raises(TypeError):
        make_split(cls, pivot_value=1.5)


@pytest.mark.parametrize("cls", [SplitCNode, PconcNode])
def test_categorical_pivot_stored_as_list(cls):
    assert make_split(cls, pivot_value=np.array([0, 3])).pivot_value == [0, 3]


@pytest.mark.parametrize("cls", [SplitNode, PlinNode, BlinNode, PconNode])
@pytest.mark.parametrize("value, left", [(1.0, True), (1.5, True), (2.0, False)])
def test_goes_left_numeric(cls, value, left):
    assert make_split(cls).goes_left(value) is left


@pytest.mark.parametrize("cls", [SplitCNode, PconcNode])
@pytest.mark.parametrize("value, left", [(0, True), (3.0, True), (1, False)])
def test_goes_left_categorical(cls, value, left):
    assert make_split(cls).goes_left(value) is left


@pytest.mark.parametrize("cls", CLASSES)
def test_round_trip_keeps_class(cls):
    node = make_split(cls)
    restored = cls.from_dict(node.to_dict())
    assert type(restored) is cls
    assert restored.pivot_idx == node.pivot_idx
    assert restored.pivot_value == node.pivot_value
    assert type(restored.left_model) is type(node.left_model)
    assert type(restored.right_model) is type(node.right_model)
    assert [child.id for child in restored.get_children()] == [1, 2]


def test_round_trip_nested():
    inner = make_split(PconNode, left=make_leaf(2), right=NoneNode())
    inner.set_id(1)
    lin = LinearNode(np.array([0]), np.array([0.0]), 0.0, 4, SimpleLinearNodeModel(4, 1.0, 0.0), make_leaf(4))
    lin.set_id(3)
    root = make_split(PlinNode, left=inner, right=lin)

    restored = PlinNode.from_dict(root.to_dict())
    left, right = restored.get_children()
    assert isinstance(left, PconNode)
    assert isinstance(left.left_child, LeafNode)
    assert isinstance(left.right_child, NoneNode)
    assert isinstance(right, LinearNode)
    assert isinstance(right.child, LeafNode)


def test_get_all_children(tree):
    assert [n.id for n in tree.get_all_children()] == [2, 4, 3]
