from types import SimpleNamespace

import numpy as np
import pytest

from imote.adapters.pilot_adapter import PilotAdapter
from imote.nodes.split_node import BlinNode, PconcNode, PconNode, PlinNode
from tests.helpers import X, y
from imote.viz_tree.viz_tree import VizTree

FIELDS = ["node_type", "model_depth", "feature_index", "split_value",
          "slope_left", "intercept_left", "slope_right", "intercept_right"]
CON = ("con", 1, -1, np.nan, 0, 0, 0, 0)


def build(*rows, pivot_values=None):
    """Fakes PILOT.tree_summary() from rows given in preorder."""
    columns = {f: np.array(col) for f, col in zip(FIELDS, zip(*rows))}
    node_ids, parent_ids, stack = [], [], []
    for i, depth in enumerate(columns["model_depth"]):
        stack = stack[:depth]
        node_ids.append(f"{depth}_{i}")
        parent_ids.append(stack[-1] if stack else None)
        stack.append(node_ids[-1])
    columns["node_id"] = np.array(node_ids, dtype=object)
    columns["parent_node_id"] = np.array(parent_ids, dtype=object)
    columns["pivot_values"] = np.array([(pivot_values or {}).get(i) for i in range(len(rows))], dtype=object)
    summary = SimpleNamespace(**columns)
    return PilotAdapter.build_root_node(X, y, SimpleNamespace(tree_summary=lambda: summary))


@pytest.mark.parametrize("root_row, cls", [
    (("pcon", 0, 0, 3.5, 0, 1.0, 0, -1.0), PconNode),
    (("blin", 0, 0, 3.5, 2.0, 0.0, -2.0, 14.0), BlinNode),
    (("plin", 0, 0, 3.5, 2.0, 0.0, -2.0, 14.0), PlinNode),
])
def test_node_type_mapping(root_row, cls):
    assert type(build(root_row, CON, CON)) is cls


def test_pconc_splits_on_categories():
    # x1 in {0}: left +1 / right -1
    root = build(("pconc", 0, 1, np.nan, 0, 1.0, 0, -1.0), CON, CON, pivot_values={0: np.array([0.0])})
    assert type(root) is PconcNode
    assert root.pivot_value == [0.0]
    np.testing.assert_array_equal(root.left_child.indices, X[:, 1] == 0)


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
