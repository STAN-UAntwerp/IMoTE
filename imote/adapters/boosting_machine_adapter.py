import os
import re
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from imote.adapters.base_adapter import BaseAdapter, register_adapter
from imote.adapters.adapter_utils import split_by_threshold
from imote.adapters.bcpilot_adapter import _working_response
from imote.nodes.base_node import BaseNode
from imote.nodes.leaf_node import LeafNode
from imote.nodes.split_node import SplitNode
from imote.nodes.node_model import LinearNodeModel, NoneNodeModel

_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_TREE_SEPARATOR = "#"
_META_CODES = ("Root", "Class0", "Class1", "Target Field")

# Factor from the summed leaf scores F to the log-odds log(p / (1 - p)), per score method.
# Exponential loss (AdaBoost) estimates F = 1/2 log(p / (1 - p)), so the log-odds are 2F.
# Score methods not listed are taken to be on the log-odds scale already.
LOGODDS_SCALE = {"Exponential Loss": 2.0}


def split_tree_path(model_path: str) -> tuple[str, int]:
    """Splits "model.xlsx#3" into ("model.xlsx", 3), a path without "#<n>" selects tree 0.

    A path that exists as given is never split, so file names containing "#" still work.
    """
    head, sep, tail = model_path.rpartition(_TREE_SEPARATOR)
    if not sep or os.path.exists(model_path) or not tail.strip().isdigit():
        return model_path, 0
    return head, int(tail)


def _read_xlsx_rows(path: str) -> list[dict]:
    """Reads the first sheet of an xlsx file as a list of {header: value} dicts, all values as str."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shared = []
        if "xl/sharedStrings.xml" in names:
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(_XLSX_NS + "si"):
                shared.append("".join(t.text or "" for t in si.iter(_XLSX_NS + "t")))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row in sheet.iter(_XLSX_NS + "row"):
        cells = {}
        for c in row.iter(_XLSX_NS + "c"):
            v = c.find(_XLSX_NS + "v")
            value = "" if v is None or v.text is None else v.text
            if c.get("t") == "s" and value:
                value = shared[int(value)]
            elif c.get("t") == "inlineStr":
                value = "".join(t.text or "" for t in c.iter(_XLSX_NS + "t"))
            cells[re.match(r"[A-Z]+", c.get("r")).group()] = value
        rows.append(cells)
    if not rows:
        raise ValueError(f"{path} has no rows")
    header = rows[0]
    return [{header[col]: value for col, value in r.items() if col in header} for r in rows[1:]]


def _build_tree_dict(rows: list[dict]) -> dict:
    """Builds the nested dict of one tree from its rows.

    Every internal node (MethodIndex) has two branches (Rank): the left one "Field <= FieldValue and not
    missing", the right one "Field > FieldValue or missing". A branch ending in a leaf also has the leaf's
    Y=1 score. Nodes are listed breadth first, so each node is the child of the oldest unfilled branch.
    """
    nodes = {}
    for r in rows:
        if r["PropertyCode"] in _META_CODES or r["PropertyValue"] == "Leaf":
            continue
        node = nodes.setdefault(int(r["MethodIndex"]), {"ranks": [], "children": [None, None]})
        rank = int(r["Rank"])
        if rank not in node["ranks"]:
            node["ranks"].append(rank)
        side = node["ranks"].index(rank)
        if r["PropertyCode"] == "Comparator" and r["PropertyValue"] == "<=":
            node["feature"], node["threshold"] = r["Field"], float(r["FieldValue"])
        elif r["PropertyCode"] == "Leaf" and r["NewField"].startswith("Y=1"):
            node["children"][side] = {"leaf": int(r["PropertyValue"]), "score": float(r["NewFieldValue"])}

    root, open_branches = None, []
    for idx in sorted(nodes):
        node = nodes[idx]
        if "feature" not in node or len(node["ranks"]) != 2:
            raise ValueError(f"Node {idx} is not a binary split")
        del node["ranks"]
        if open_branches:
            parent, side = open_branches.pop(0)
            parent["children"][side] = node
        elif root is None:
            root = node
        else:
            raise ValueError(f"Node {idx} has no parent")
        open_branches += [(node, side) for side in (0, 1) if node["children"][side] is None]
    if root is None or open_branches:
        raise ValueError("Tree has branches without a node or leaf")
    return root


def _tree_features(tree: dict) -> set[str]:
    if "leaf" in tree:
        return set()
    return {tree["feature"]} | _tree_features(tree["children"][0]) | _tree_features(tree["children"][1])


def _tree_scores(tree: dict, X: np.ndarray, feature_idx: dict[str, int]) -> np.ndarray:
    """Leaf score of the tree for every row of X, missing values (NaN) go right."""
    scores = np.empty(X.shape[0])
    stack = [(tree, np.ones(X.shape[0], dtype=bool))]
    while stack:
        node, rows = stack.pop()
        if "leaf" in node:
            scores[rows] = node["score"]
            continue
        left_rows, right_rows = split_by_threshold(X, rows, feature_idx[node["feature"]], node["threshold"])
        stack += [(node["children"][0], left_rows), (node["children"][1], right_rows)]
    return scores


@register_adapter("BoostingMachine")
class BoostingMachineAdapter(BaseAdapter):
    """Adapter for one tree of a binary classification BoostingMachine, exported as an xlsx rule table.

    The model path selects the tree: "model.xlsx" loads tree 0, "model.xlsx#3" loads tree 3.
    The tree is shown as a step of the ensemble: the log_odds of a node are the log-odds of its samples
    after the trees before it, a leaf adds its score, and y_hat are the log-odds after the shown tree.
    Splits refer to features by name, so the dataset needs columns with the same names and a 0/1 target.
    """

    task = "classification"
    needs_feature_names = True
    label = "BoostingMachine (Amadea)"

    @staticmethod
    def load_model(model_path: str) -> dict:
        """Loads the trees of a BoostingMachine xlsx export and selects the tree given after "#".

        Args:
            model_path: Path to the xlsx file, optionally followed by "#<tree number>".

        Returns:
            Dict with the selected "tree_index", all "trees" in order and the "logodds_scale".

        Raises:
            FileNotFoundError: If the xlsx file does not exist.
            ValueError: If the file is not a BoostingMachine export or the tree number does not exist.
        """
        path, tree_index = split_tree_path(model_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file path to load model: {path}")
        try:
            rows = _read_xlsx_rows(path)
        except (zipfile.BadZipFile, KeyError) as e:
            raise ValueError(f"{path} is not a readable xlsx file ({e})") from e
        if not rows or rows[0].get("MethodID") != "BoostingMachine":
            raise ValueError(f"{path} is not a BoostingMachine export")

        rows_per_tree = {}
        score_method = None
        for r in rows:
            if r.get("AggregationValues"):
                rows_per_tree.setdefault(r["AggregationValues"], []).append(r)
            elif r.get("PropertyCode") == "ScoreMethod":
                score_method = r["PropertyValue"]
        tree_ids = sorted(rows_per_tree, key=int)
        if tree_index >= len(tree_ids):
            raise ValueError(f"Tree {tree_index} does not exist, the model has trees 0 to {len(tree_ids) - 1}")

        return {
            "tree_index": tree_index,
            "trees": [_build_tree_dict(rows_per_tree[t]) for t in tree_ids[:tree_index + 1]],
            "n_trees": len(tree_ids),
            "logodds_scale": LOGODDS_SCALE.get(score_method, 1.0),
        }

    @staticmethod
    def _feature_idx(model: dict, feature_names) -> dict[str, int]:
        needed = set().union(*(_tree_features(t) for t in model["trees"]))
        if feature_names is None:
            raise ValueError("The BoostingMachine adapter needs a dataset with feature names")
        missing = needed - set(feature_names)
        if missing:
            raise ValueError(f"The dataset has no column(s) {', '.join(sorted(missing))}")
        return {name: list(feature_names).index(name) for name in needed}

    @staticmethod
    def _logodds(model: dict, X: np.ndarray, feature_idx: dict[str, int], n_trees: int) -> np.ndarray:
        """Log-odds of the first n_trees trees for every row of X."""
        f = np.zeros(X.shape[0])
        for tree in model["trees"][:n_trees]:
            f += _tree_scores(tree, X, feature_idx)
        return model["logodds_scale"] * f

    @staticmethod
    def build_root_node(X_train, y_train, model, feature_names=None) -> BaseNode:
        if not np.isin(y_train, (0, 1)).all():
            raise ValueError("The BoostingMachine adapter needs a 0/1 target")
        X_train = X_train.astype(float)
        feature_idx = BoostingMachineAdapter._feature_idx(model, feature_names)
        f_before = BoostingMachineAdapter._logodds(model, X_train, feature_idx, model["tree_index"])
        root_indices = np.ones(X_train.shape[0], dtype=bool)
        return BoostingMachineAdapter._build_recursive(
            model["trees"][model["tree_index"]], model["logodds_scale"], feature_idx,
            X_train, y_train, root_indices, f_before,
        )

    @staticmethod
    def _build_recursive(tree, scale, feature_idx, X_train, y_train, current_indices, f_before) -> BaseNode:
        f = f_before[current_indices]
        y_res = _working_response(y_train[current_indices], f)
        rss = float(np.sum(y_res ** 2))

        if "leaf" in tree:
            node = LeafNode(
                indices=current_indices,
                y_res=y_res,
                rss=rss,
                node_model=LinearNodeModel(np.zeros(X_train.shape[1]), scale * tree["score"]),
            )
        else:
            pivot_idx = feature_idx[tree["feature"]]
            left_indices, right_indices = split_by_threshold(X_train, current_indices, pivot_idx, tree["threshold"])
            node = SplitNode(
                indices=current_indices,
                y_res=y_res,
                rss=rss,
                pivot_idx=pivot_idx,
                pivot_value=tree["threshold"],
                left_child=BoostingMachineAdapter._build_recursive(
                    tree["children"][0], scale, feature_idx, X_train, y_train, left_indices, f_before),
                right_child=BoostingMachineAdapter._build_recursive(
                    tree["children"][1], scale, feature_idx, X_train, y_train, right_indices, f_before),
                left_model=NoneNodeModel(),
                right_model=NoneNodeModel(),
            )
        node.log_odds = f
        return node

    @staticmethod
    def predict(X, model, feature_names=None) -> np.ndarray:
        feature_idx = BoostingMachineAdapter._feature_idx(model, feature_names)
        return BoostingMachineAdapter._logodds(model, X.astype(float), feature_idx, model["tree_index"] + 1)
