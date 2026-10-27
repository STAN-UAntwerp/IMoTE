from imote.nodes.label_formatter import LabelFormatter
from imote.nodes.none_node import NoneNode


def test_has_no_children():
    assert NoneNode().get_children() == []


def test_labels_are_strings():
    node = NoneNode()
    assert isinstance(node.get_label(LabelFormatter()), str)
    assert isinstance(node.get_minimal_label(LabelFormatter()), str)


def test_round_trip():
    node = NoneNode.from_dict(NoneNode().to_dict())
    assert isinstance(node, NoneNode)
    assert node.id == -1
