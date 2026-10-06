import json

import numpy as np
import pytest

from imote.adapters.partykit_adapter import PartyKitAdapter
from imote.nodes.leaf_node import LeafNode
from imote.nodes.split_node import SplitCNode, SplitNode
from tests.helpers import X, y
from imote.viz_tree.viz_tree import VizTree

# root: x0 <= 3.5
# ├── x1 in {0, 2}                   (partykit index: 1 = left kid, 2 = right kid)
# │   ├── leaf x0 + 1                rows 0, 2, 3
# │   └── leaf 3*x2 + 2              row 1
# └── leaf 2*x2                      rows 4-7
PARTYKIT_MODEL = {
    "names": ["y", "x0", "x1", "x2"],
    "nodes": [
        {"is_terminal": False, "split_var": "x0", "breaks": 3.5, "kids": [1, 4]},
        {"is_terminal": False, "split_var": "x1", "index": [1, 2, 1], "levels": [0, 1, 2], "kids": [2, 3]},
        {"is_terminal": True, "coefficients": {"(Intercept)": 1.0, "x0": 1.0}},
        {"is_terminal": True, "coefficients": {"(Intercept)": 2.0, "x1": 0.0, "x2": 3.0}},
        {"is_terminal": True, "coefficients": {"(Intercept)": 0.0, "x2": 2.0}},
    ],
}


@pytest.fixture
def tree():
    return VizTree.from_model(PartyKitAdapter, X, y, PARTYKIT_MODEL)


def test_structure(tree):
    assert [type(n) for n in tree.nodes] == [SplitNode, SplitCNode, LeafNode, LeafNode, LeafNode]
    root, split_c = tree.nodes[0], tree.nodes[1]
    assert (root.pivot_idx, split_c.pivot_idx) == (0, 1)
    assert list(split_c.pivot_value) == [0, 2]


def test_leaf_coefficients_follow_column_order(tree):
    # missing feature names get coefficient 0
    np.testing.assert_array_equal(tree.nodes[3].node_model.coefficients, [1, 0, 0])
    np.testing.assert_array_equal(tree.nodes[4].node_model.coefficients, [0, 0, 3])
    assert tree.nodes[4].node_model.intercept == 2.0


def test_predict(tree):
    np.testing.assert_allclose(tree.predict(X), [1, 5, 3, 4, 0.2, 1.6, 0.6, 1.4])


def test_load_model(tmp_path):
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(PARTYKIT_MODEL))
    assert PartyKitAdapter.load_model(str(path)) == PARTYKIT_MODEL
