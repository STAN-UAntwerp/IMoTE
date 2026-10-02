from types import SimpleNamespace

import numpy as np
import pytest

from adapters.pilotc_adapter import PilotCAdapter
from nodes.internal_node import LinearNode
from nodes.leaf_node import LeafNode
from nodes.split_node import BlinNode, PconNode, PlinNode
from tests.helpers import X, y
from viz_tree.viz_tree import VizTree

FIELDS = ["node_type", "model_depth", "feature_index", "split_value",
          "slope_left", "intercept_left", "slope_right", "intercept_right"]
CON = ("con", 1, -1, np.nan, 0, 0, 0, 0)


def build(*rows):
    """Rows in preorder; children are the next nodes with model_depth + 1."""
    summary = SimpleNamespace(**{f: np.array(col) for f, col in zip(FIELDS, zip(*rows))})
    return PilotCAdapter.build_root_node(X, y, SimpleNamespace(tree_summary=lambda: summary))


@pytest.mark.parametrize("root_row, cls", [
    (("con", 0, -1, np.nan, 0, 0, 0, 0), LeafNode),
    (("lin", 0, 2, np.nan, 1.0, 0.0, 0, 0), LinearNode),
    (("pcon", 0, 0, 3.5, 0, 1.0, 0, -1.0), PconNode),
    (("blin", 0, 0, 3.5, 2.0, 0.0, -2.0, 14.0), BlinNode),
    (("plin", 0, 0, 3.5, 2.0, 0.0, -2.0, 14.0), PlinNode),
])
def test_node_type_mapping(root_row, cls):
    assert type(build(root_row, CON, CON)) is cls


def test_pconc_not_implemented():
    with pytest.raises(NotImplementedError):
        build(("pconc", 0, 1, 0, 0, 1.0, 0, -1.0), CON, CON)


def test_leaf_model_accumulates_path():
    # lin(x2: 1*x2) -> pcon(x0 <= 3.5: left +1 / right -1) -> two leaves
    root = build(("lin", 0, 2, np.nan, 1.0, 0.0, 0, 0),
                 ("pcon", 1, 0, 3.5, 0, 1.0, 0, -1.0),
                 ("con", 2, -1, np.nan, 0, 0, 0, 0),
                 ("con", 2, -1, np.nan, 0, 0, 0, 0))
    left = root.child.left_child
    np.testing.assert_array_equal(left.node_model.coefficients, [0, 0, 1])
    assert left.node_model.intercept == 1.0
    tree = VizTree(root, X, y, None)
    np.testing.assert_allclose(tree.predict(X), np.where(X[:, 0] <= 3.5, X[:, 2] + 1, X[:, 2] - 1))
