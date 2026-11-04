import zipfile
from xml.sax.saxutils import escape

import numpy as np
import pytest

from imote.adapters.boosting_machine_adapter import BoostingMachineAdapter, split_tree_path
from imote.nodes.leaf_node import LeafNode
from imote.nodes.split_node import SplitNode
from imote.viz_tree.viz_tree import VizTree

HEADER = ["ModelGroup", "MethodIndex", "MethodID", "AggregationFields", "AggregationValues", "Rank", "Field",
          "FieldValue", "NewField", "NewFieldType", "NewFieldValue", "PropertyCode", "PropertyValue"]
FEATURE_NAMES = ["a", "b"]


def _split(tree, idx, rank, field, threshold, left_score=None, right_score=None):
    """Rows of one split as the BoostingMachine exports it, a score adds a leaf to that branch."""
    rows = []
    for side, (cmp, missing, op, score) in enumerate((("<=", "<>", "&&", left_score), (">", "=", "||", right_score))):
        base = ["M", idx, "BoostingMachine", "", tree, rank + side]
        rows += [base + [field, threshold, "", "", "", "Comparator", cmp],
                 base + [field, "", "", "", "", "Comparator", missing],
                 base + [field, "", "", "", side, "Operator", op]]
        if score is not None:
            rows += [base + ["", "", "Y=0 ProbaEstimate", "real", -score, "Leaf", rank + side],
                     base + ["", "", "Y=1 ProbaEstimate", "real", score, "Leaf", rank + side],
                     base + ["", "", "Y Predicted Label", "integer", int(score > 0), "Label", 0]]
    return [[str(v) for v in r] for r in rows]


# tree 00 (breadth first, MethodIndex counts leaves too):
# [0] a <= 2
# ├── [1] b <= 0.5 -> leaf +1 / leaf -1
# └── leaf +0.5
# tree 01:
# [0] b <= 0.5 -> leaf -0.25 / leaf +0.25
ROWS = (_split("00", 0, 1, "a", 2, right_score=0.5)
        + _split("00", 1, 3, "b", 0.5, left_score=1.0, right_score=-1.0)
        + [["M", "4", "BoostingMachine", "", "00", "5", "", "", "", "", "", "Root", "-1"]]
        + _split("01", 0, 1, "b", 0.5, left_score=-0.25, right_score=0.25)
        + [["M", "", "BoostingMachine", "", "", "", "", "", "Y=1 ProbaEstimate", "", "1", "ScoreMethod", "Exponential Loss"]])

X = np.array([[1, 0], [1, 1], [3, 0], [3, 1], [1, np.nan]], dtype=float)
y = np.array([1, 0, 1, 0, 0], dtype=float)


def _write_xlsx(path, rows):
    def cell(col, value):
        return f'<c r="{col}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'

    sheet_rows = []
    for r, values in enumerate([HEADER] + rows, start=1):
        cells = "".join(cell(f"{chr(65 + c)}{r}", v) for c, v in enumerate(values) if v != "")
        sheet_rows.append(f'<row r="{r}">{cells}</row>')
    sheet = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "trees.xlsx"
    _write_xlsx(path, ROWS)
    return str(path)


def test_split_tree_path(tmp_path):
    assert split_tree_path("m.xlsx") == ("m.xlsx", 0)
    assert split_tree_path("m.xlsx#3") == ("m.xlsx", 3)
    assert split_tree_path("m.xlsx#x") == ("m.xlsx#x", 0)
    existing = tmp_path / "m#1"
    existing.write_text("")
    assert split_tree_path(str(existing)) == (str(existing), 0)


def test_structure_first_tree(model_path):
    tree = VizTree.from_model(BoostingMachineAdapter, X, y, BoostingMachineAdapter.load_model(model_path), FEATURE_NAMES)
    assert tree.task == "classification"
    assert [type(n) for n in tree.nodes] == [SplitNode, SplitNode, LeafNode, LeafNode, LeafNode]
    root, inner = tree.nodes[0], tree.nodes[1]
    assert (root.pivot_idx, root.pivot_value, inner.pivot_idx, inner.pivot_value) == (0, 2.0, 1, 0.5)
    # exponential loss: log-odds are twice the summed leaf scores, the missing b of row 4 goes right
    np.testing.assert_allclose(tree.y_hat, [2, -2, 1, 1, -2])
    np.testing.assert_allclose(root.log_odds, np.zeros(5))


def test_second_tree_starts_from_first(model_path):
    model = BoostingMachineAdapter.load_model(model_path + "#1")
    tree = VizTree.from_model(BoostingMachineAdapter, X, y, model, FEATURE_NAMES)
    assert [type(n) for n in tree.nodes] == [SplitNode, LeafNode, LeafNode]
    np.testing.assert_allclose(tree.root_node.log_odds, [2, -2, 1, 1, -2])
    np.testing.assert_allclose(tree.y_hat, [1.5, -1.5, 0.5, 1.5, -1.5])


def test_load_errors(model_path, tmp_path):
    with pytest.raises(ValueError, match="trees 0 to 1"):
        BoostingMachineAdapter.load_model(model_path + "#2")
    with pytest.raises(FileNotFoundError):
        BoostingMachineAdapter.load_model(str(tmp_path / "missing.xlsx"))
    with pytest.raises(ValueError, match="no column"):
        VizTree.from_model(BoostingMachineAdapter, X, y, BoostingMachineAdapter.load_model(model_path), ["a", "c"])
